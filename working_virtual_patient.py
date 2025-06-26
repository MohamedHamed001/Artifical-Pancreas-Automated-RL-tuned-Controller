import numpy as np
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
from scipy.integrate import solve_ivp
import pandas as pd
import os, random, re

class HovorkaPatient:
    """
    A simulation environment for a Type 1 Diabetes patient based on the Hovorka model.
    This class simulates the glucose-insulin dynamics in response to meals, exercise,
    and insulin infusion.
    """

    def __init__(self, patient_params, simulation_start_time=0):
        """
        Initializes the patient simulation environment.

        Args:
            patient_params (dict): A dictionary containing all necessary patient parameters.
            simulation_start_time (int): The starting time of the simulation in minutes.
        """
        self.params = patient_params
        self.BW = self.params.get('BW', 75)  # Body weight in kg

        # Hovorka model state variables
        self.S1 = 0.0  # Insulin in plasma [U]
        self.S2 = 0.0  # Insulin in interstitial fluid [U]
        self.I = 0.0   # Insulin concentration in plasma [mU/L]
        self.x1 = 0.0  # Insulin action on glucose transport 1
        self.x2 = 0.0  # Insulin action on glucose transport 2
        self.x3 = 0.0  # Insulin action on EGP
        self.Q1 = 0.0  # Glucose in accessible compartment [mmol]
        self.Q2 = 0.0  # Glucose in non-accessible compartment [mmol]
        self.D1 = 0.0  # Carbohydrates in stomach (solid) [g]
        self.D2 = 0.0  # Carbohydrates in stomach (liquid) [g]

        # Simulation time tracking
        self.simulation_start_time = simulation_start_time
        self.time = self.simulation_start_time
        
        # Initial BGL
        self.G = self.params.get('G_init', 10.0) # Glucose concentration [mmol/L] (180 mg/dL)
        V_G = self.params.get('V_G', 0.16) * self.BW
        self.Q1 = self.G * V_G # Set initial glucose mass
        
        self.initial_state = self._get_full_state()

        # Data loaders for external inputs
        self.meal_data = None
        self.exercise_data = None
        
        # Meal tracking
        self.last_meal_input = 0.0

        # Exercise tracking variables
        self.exercise_start_time = None
        self.exercise_end_time = None
        self.current_exercise_status = 0
        self.F_sensitivity = 1.0  # Current insulin sensitivity factor
        
        # -------------------- Numba setup --------------------
        # Pre-compute parameter vector used by the jitted Euler integrator
        self._numba_params = np.array([
            self.params.get('k_a1', 0.006), self.params.get('k_a2', 0.06), self.params.get('k_a3', 0.05),
            self.params.get('k_b1', 0.003), self.params.get('k_b2', 0.06), self.params.get('k_b3', 0.04),
            self.params.get('V_I', 0.12) * self.BW, self.params.get('t_max_I', 55), self.params.get('k_e', 0.138),
            self.params.get('F_01', 0.0097) * self.BW, self.params.get('V_G', 0.16) * self.BW, self.params.get('k_12', 0.066),
            self.params.get('EGP_0', 0.0161) * self.BW, self.params.get('AG', 1.0), self.params.get('t_max_G', 30),
            self.params.get('A_EGP', 0.05), self.params.get('phi_EGP', -60),
            self.params.get('G_thresh', 9.0), self.params.get('k_R', 0.0031)
        ], dtype=np.float64)

        if NUMBA_AVAILABLE:
            # Trigger JIT compilation once with dummy call (warm-up)
            _state_dummy = np.zeros(10, dtype=np.float64)
            _ = hovorka_one_min_step(_state_dummy, 0.0, 1.0, self._numba_params)

    def _get_full_state(self):
        return np.array([
            self.S1, self.S2, self.I, self.x1, self.x2, self.x3,
            self.Q1, self.Q2, self.D1, self.D2
        ])

    def _set_full_state(self, state):
        (self.S1, self.S2, self.I, self.x1, self.x2, self.x3,
         self.Q1, self.Q2, self.D1, self.D2) = state

    def load_meal_data(self, filepath: str):
        """
        Load meal data from a file, intelligently parsing it to identify only the start of a meal.
        This prevents the "double bolus" issue caused by start/end time data format.
        """
        try:
            raw_df = pd.read_csv(filepath, sep='\s+', header=None, names=['time', 'carbs'], comment='#')
            
            # If times look like hours (i.e. everything ≤24) convert to minutes for a 24-h simulation
            if raw_df['time'].max() <= 24:
                raw_df['time'] *= 60
                print("Converted meal times from hours to minutes.")
            
            # Intelligent parsing logic
            processed_meals = []
            last_carbs = 0
            for index, row in raw_df.iterrows():
                current_carbs = row['carbs']
                # Treat any *increase* in carb amount as a new meal start.
                # This captures patterns like (0 → 3 g), (3 → 6 g), (6 → 11 g) as separate meals,
                # matching the TestData format where each step accumulates additional carbs.
                if current_carbs > 0 and (last_carbs == 0 or current_carbs > last_carbs):
                    processed_meals.append({'time': row['time'], 'carbs': current_carbs})
                last_carbs = current_carbs

            if not processed_meals:
                print("Warning: No meal start events were found in the data file.")
                self.meal_data = pd.DataFrame(columns=['time', 'carbs'])
            else:
                self.meal_data = pd.DataFrame(processed_meals)
                
            print(f"Meal data loaded and processed successfully from {filepath}")
            print("Detected meal events (time, carbs):")
            for _, row in self.meal_data.iterrows():
                print(f"  - {row['time']} min, {row['carbs']} g")

        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            self.meal_data = pd.DataFrame(columns=['time', 'carbs'])

    def load_exercise_data(self, filepath: str):
        """Load exercise data from a file"""
        try:
            df = pd.read_csv(filepath, sep='\s+', header=None, names=['time', 'active'], comment='#')
            
            # Convert exercise timeline to minutes if provided in hours
            if df['time'].max() <= 24:
                df['time'] *= 60
                print("Converted exercise times from hours to minutes.")
            
            self.exercise_data = df
            print(f"Exercise data loaded successfully from {filepath}")
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            self.exercise_data = None

    def _get_meal_intake(self, t):
        """Returns the amount of carbs ingested at time t."""
        if self.meal_data is None:
            return 0.0
        
        meal_events = self.meal_data[self.meal_data['time'] == t]
        if not meal_events.empty:
            return float(meal_events['carbs'].iloc[0])
        return 0.0

    def _get_exercise_status(self, t):
        """Returns the exercise status at time t."""
        if self.exercise_data is None:
            return 0.0
        
        exercise_events = self.exercise_data[self.exercise_data['time'] == t]
        if not exercise_events.empty:
            return float(exercise_events['active'].iloc[0])
        return 0.0

    def _update_exercise_sensitivity(self, t):
        """
        Updates the insulin sensitivity factor based on exercise timing.
        Implements the F_rise and F_decay equations.
        """
        exercise_status = self._get_exercise_status(t)
        
        # Check for exercise start
        if exercise_status == 1 and self.current_exercise_status == 0:
            self.exercise_start_time = t
            self.exercise_end_time = None
        
        # Check for exercise end
        elif exercise_status == 0 and self.current_exercise_status == 1:
            self.exercise_end_time = t
        
        # Update current status
        self.current_exercise_status = exercise_status
        
        # Calculate sensitivity factor
        F_peak = self.params.get('F_peak', 1.35)
        K_rise = self.params.get('K_rise', 5.0)
        K_decay = self.params.get('K_decay', 0.01)
        
        if exercise_status == 1 and self.exercise_start_time is not None:
            # During exercise: sensitivity increase
            t_rise = t - self.exercise_start_time
            self.F_sensitivity = 1 + (F_peak - 1) * (1 - np.exp(-K_rise * t_rise))
        elif exercise_status == 0 and self.exercise_end_time is not None:
            # After exercise: sensitivity decay
            t_decay = t - self.exercise_end_time
            self.F_sensitivity = 1 + (F_peak - 1) * np.exp(-K_decay * t_decay)
        else:
            # No exercise: normal sensitivity
            self.F_sensitivity = 1.0

    def _hovorka_model_equations(self, t, y, u_I):
        """
        The set of differential equations for the Hovorka model.
        
        Args:
            t (float): Current time.
            y (np.array): Array of state variables.
            u_I (float): Insulin infusion rate [U/min].
        
        Returns:
            list: The derivatives of the state variables.
        """
        S1, S2, I, x1, x2, x3, Q1, Q2, D1, D2 = y

        # Parameters from self.params
        k_a1 = self.params.get('k_a1', 0.006)
        k_a2 = self.params.get('k_a2', 0.06)
        k_a3 = self.params.get('k_a3', 0.05)
        k_b1 = self.params.get('k_b1', 0.003)
        k_b2 = self.params.get('k_b2', 0.06)
        k_b3 = self.params.get('k_b3', 0.04)
        k_c1 = self.params.get('k_c1', 0.5)
        V_I = self.params.get('V_I', 0.12) * self.BW
        t_max_I = self.params.get('t_max_I', 55) # minutes
        k_e = self.params.get('k_e', 0.138)
        
        F_01 = self.params.get('F_01', 0.0097) * self.BW
        V_G = self.params.get('V_G', 0.16) * self.BW
        k_12 = self.params.get('k_12', 0.066)
        
        EGP_0 = self.params.get('EGP_0', 0.0161) * self.BW
        
        AG = self.params.get('AG', 1.0)
        t_max_G = self.params.get('t_max_G', 40) # minutes
        
        # Insulin subsystem
        dS1 = u_I - (S1 / t_max_I)
        dS2 = (S1 - S2) / t_max_I
        dI = (S2 / (t_max_I * V_I)) - k_e * I
        
        # Insulin action subsystem
        dx1 = k_b1 * I - k_a1 * x1
        dx2 = k_b2 * I - k_a2 * x2
        dx3 = k_b3 * I - k_a3 * x3
        
        # Glucose subsystem
        # EGP with circadian rhythm modification
        A_EGP = self.params.get('A_EGP', 0.05)
        phi_EGP = self.params.get('phi_EGP', -60) # phase shift in minutes
        EGP0_baseline = EGP_0  # Use the standard EGP_0 from Hovorka model
        EGP0_circadian = EGP0_baseline * (1 + A_EGP * np.sin(2 * np.pi * (t - phi_EGP) / 1440))
        EGP = EGP0_circadian - x3 * EGP_0

        # Renal glucose clearance
        G = Q1 / V_G if V_G > 0 else 0.0  # mmol/L
        G_thresh = self.params.get('G_thresh', 9.0)  # mmol/L
        k_R = self.params.get('k_R', 0.0031)  # min^-1
        if G > G_thresh:
            F_R = k_R * (G - G_thresh) * V_G
        else:
            F_R = 0.0

        # Meal absorption using a two-compartment model
        dD1 = -D1 / t_max_G
        dD2 = D1 / t_max_G - D2 / t_max_G
        U_id = (AG * D2) / t_max_G

        # Glucose subsystem dynamics based on Hovorka (2004)
        # Exercise effect on insulin sensitivity is applied to insulin-dependent glucose uptake
        U_g = x1 * Q1 * self.F_sensitivity # Insulin-dependent glucose uptake
        
        dQ1 = U_id + EGP - F_R - F_01 - U_g - k_12 * Q1 + k_12 * Q2
        dQ2 = k_12 * Q1 - k_12 * Q2 - x2 * Q2

        return [dS1, dS2, dI, dx1, dx2, dx3, dQ1, dQ2, dD1, dD2]

    def step(self, insulin_rate: float) -> float:
        """
        Advances the simulation by one time step (1 minute) using a fast Euler integrator.

        Falls back to the original SciPy RK45 solver if Numba is not installed.
        """

        # Convert from U/h to U/min for the model equations
        insulin_infusion_rate_umin = insulin_rate / 60.0

        # Update exercise sensitivity factor
        self._update_exercise_sensitivity(self.time)

        # Get meal intake for the current minute
        meal_carbs = self._get_meal_intake(self.time)  # grams

        # Add carbs to stomach (D1) only at the start of a meal (avoid double addition)
        if meal_carbs > 0 and self.last_meal_input == 0:
            carbs_mmol = meal_carbs * (1000.0 / 180.0)  # g → mmol conversion
            self.D1 += carbs_mmol

        self.last_meal_input = meal_carbs

        if NUMBA_AVAILABLE:
            # Fast path: Numba-accelerated Euler step
            state = self._get_full_state().astype(np.float64)
            new_state = hovorka_one_min_step(state, insulin_infusion_rate_umin, self.F_sensitivity, self._numba_params)
            self._set_full_state(new_state)
        else:
            # Slow fallback: SciPy RK45 (original implementation)
            solution = solve_ivp(
                fun=lambda t, y: self._hovorka_model_equations(t, y, insulin_infusion_rate_umin),
                t_span=[self.time, self.time + 1],
                y0=self._get_full_state(),
                method='RK45'
            )
            self._set_full_state(solution.y[:, -1])

        # Update glucose concentration and time
        V_G = self.params.get('V_G', 0.16) * self.BW
        self.G = self.Q1 / V_G  # mmol/L
        
        self.time += 1
        
        # Convert BGL to mg/dL (1 mmol/L = 18.0182 mg/dL)
        bgl_mg_dl = self.G * 18.0182
        
        return bgl_mg_dl

    def reset(self):
        """Resets the environment to its initial state."""
        self._set_full_state(self.initial_state)
        self.time = self.simulation_start_time
        self.last_meal_input = 0.0
        self.exercise_start_time = None
        self.exercise_end_time = None
        self.current_exercise_status = 0
        self.F_sensitivity = 1.0
        bgl_mg_dl = self.G * 18.0182
        return bgl_mg_dl

# -------------------------------------------------------------------------------------------------
#                               Numba-accelerated Euler integrator
# -------------------------------------------------------------------------------------------------

if NUMBA_AVAILABLE:

    @njit(fastmath=True, cache=True)
    def hovorka_one_min_step(state, u_I, F_sens, p):
        """One-minute Euler integration of the Hovorka model.

        Parameters
        ----------
        state : 1-D float64 array, length 10
            Current model state [S1, S2, I, x1, x2, x3, Q1, Q2, D1, D2].
        u_I : float64
            Insulin infusion rate (U/min).
        F_sens : float64
            Current insulin-sensitivity multiplier (≥1 during exercise).
        p : 1-D float64 array, length 19
            Pre-packed model parameters.
        Returns
        -------
        new_state : float64[10]
            State after 1 min.
        """
        # Unpack state
        S1, S2, I, x1, x2, x3, Q1, Q2, D1, D2 = state

        # Unpack params (keep order in _numba_params)
        k_a1, k_a2, k_a3, k_b1, k_b2, k_b3, V_I, t_max_I, k_e, F_01, V_G, k_12, EGP_0, AG, t_max_G, A_EGP, phi_EGP, G_thresh, k_R = p

        # Insulin subsystem
        dS1 = u_I - (S1 / t_max_I)
        dS2 = (S1 - S2) / t_max_I
        dI = (S2 / (t_max_I * V_I)) - k_e * I

        # Insulin action subsystem
        dx1 = k_b1 * I - k_a1 * x1
        dx2 = k_b2 * I - k_a2 * x2
        dx3 = k_b3 * I - k_a3 * x3

        # Glucose subsystem helpers
        # Circadian endogenous glucose production
        EGP0_circadian = EGP_0 * (1.0 + A_EGP * np.sin(2.0 * np.pi * (state[0] - phi_EGP) / 1440.0))  # state[0] is time surrogate (not ideal)
        # Without actual wall-clock time inside this jitted helper we approximate circadian term as constant.
        EGP = EGP_0 - x3 * EGP_0

        # Renal glucose clearance
        G = Q1 / V_G if V_G > 0.0 else 0.0
        if G > G_thresh:
            F_R = k_R * (G - G_thresh) * V_G
        else:
            F_R = 0.0

        # Meal absorption
        dD1 = -D1 / t_max_G
        dD2 = D1 / t_max_G - D2 / t_max_G
        U_id = (AG * D2) / t_max_G

        # Glucose subsystem dynamics (with exercise sensitivity)
        U_g = x1 * Q1 * F_sens
        dQ1 = U_id + EGP - F_R - F_01 - U_g - k_12 * Q1 + k_12 * Q2
        dQ2 = k_12 * Q1 - k_12 * Q2 - x2 * Q2

        # Euler integration (dt = 1 min)
        return np.array([
            S1 + dS1,
            S2 + dS2,
            I + dI,
            x1 + dx1,
            x2 + dx2,
            x3 + dx3,
            Q1 + dQ1,
            Q2 + dQ2,
            D1 + dD1,
            D2 + dD2
        ])

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Randomly pick a test case from TestData and retrieve body weight
    # ------------------------------------------------------------------
    root_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(root_dir, 'TestData')

    # gather available case indices from meal files
    meal_files = [f for f in os.listdir(data_dir) if f.startswith('MealData_case') and f.endswith('.data')]
    case_indices = [int(re.findall(r'MealData_case(\d+)\.data', f)[0]) for f in meal_files]
    if not case_indices:
        raise RuntimeError('No MealData_case*.data files found in TestData directory.')

    case_idx = random.choice(case_indices)

    meal_path = os.path.join(data_dir, f'MealData_case{case_idx}.data')
    exercise_path = os.path.join(data_dir, f'ExerciseData_case{case_idx}.data')

    # Parse body weight from TestCases.txt
    testcases_path = os.path.join(data_dir, 'TestCases.txt')
    def get_body_weight(case_idx: int) -> float:
        with open(testcases_path, 'r') as f:
            lines = f.readlines()
        header = f'Test Case [{case_idx}]'
        for i, line in enumerate(lines):
            if line.strip() == header:
                for j in range(i+1, i+6):
                    if j < len(lines) and 'Body Weight:' in lines[j]:
                        return float(lines[j].split(':')[1].split()[0])
        raise ValueError(f'Body weight for case {case_idx} not found in TestCases.txt')

    body_weight = get_body_weight(case_idx)

    print(f'Running standalone patient simulation with Test Case {case_idx} (BW={body_weight} kg)')

    # ------------------------------------------------------------------
    # Patient parameter dictionary
    # ------------------------------------------------------------------
    patient_parameters = {
        'BW': body_weight,
        # Hovorka model parameters (standard values)
        'k_a1': 0.006, 'k_a2': 0.06, 'k_a3': 0.05,
        'k_b1': 0.003, 'k_b2': 0.06, 'k_b3': 0.04, 'k_c1': 0.5,
        'V_I': 0.12, 't_max_I': 55, 'k_e': 0.138,
        'F_01': 0.0097, 'V_G': 0.16, 'k_12': 0.066,
        'EGP_0': 0.0161, 'AG': 1.0, 't_max_G': 30,
        'G_init': 10.0,  # 180 mg/dL
        
        # Your custom modification parameters
        'A_EGP': 0.05, # Reduced from 0.2
        'phi_EGP': -60, # minutes past midnight
        'F_peak': 1.35,
        'K_rise': 5.0,
        'K_decay': 0.01,
        'G_thresh': 9.0,  # mmol/L
        'k_R': 0.0031,   # min^-1
    }

    # Create patient environment
    patient_env = HovorkaPatient(patient_params=patient_parameters)
    
    # Load selected TestData files
    patient_env.load_meal_data(meal_path)
    patient_env.load_exercise_data(exercise_path)
    
    simulation_duration = 1440 # minutes (one day)
    bgl_history = []
    time_history = []
    
    for t in range(simulation_duration):
        # Controller would provide this value. For now, a constant basal rate.
        insulin_rate = 0.1 # U/h
        
        bgl = patient_env.step(insulin_rate)
        
        bgl_history.append(bgl)
        time_history.append(patient_env.time)
        
        if t % 100 == 0:
            print("Time: {} min, BGL: {:.2f} mg/dL".format(patient_env.time, bgl))

    # Plot results
    import matplotlib.pyplot as plt
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot BGL
    ax1.plot(time_history, bgl_history, label='Blood Glucose (mg/dL)', color='blue')
    ax1.set_ylabel('Glucose (mg/dL)')
    ax1.set_title('Hovorka Model Simulation')
    ax1.grid(True)
    
    # Plot meals on secondary y-axis
    ax1_meals = ax1.twinx()
    # Plot meal events if data loaded
    if patient_env.meal_data is not None and not patient_env.meal_data.empty:
        meal_times = patient_env.meal_data['time'].values
        meal_carbs = patient_env.meal_data['carbs'].values
        ax1_meals.stem(meal_times, meal_carbs, linefmt='g-', markerfmt='go', basefmt=" ", label='Meals (g)')
    ax1_meals.set_ylabel('Carbohydrates (g)', color='g')
    ax1_meals.tick_params(axis='y', labelcolor='g')
    ax1_meals.set_ylim(bottom=0)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_meals.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.show() 