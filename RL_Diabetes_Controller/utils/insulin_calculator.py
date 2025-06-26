import numpy as np

class InsulinCalculator:
    """
    Calculates insulin doses for diabetes management including meal bolus and correction doses.
    """
    
    def __init__(self, patient_weight_kg=75):
        """
        Initialize insulin calculator.
        
        Args:
            patient_weight_kg (float): Patient weight in kilograms
        """
        self.patient_weight = patient_weight_kg
        self.tdi = self._calculate_tdi()
        self.carb_ratio = self._calculate_carb_ratio()
        self.isf = self._calculate_isf()
        
        # Insulin delivery tracking
        self.last_insulin_time = 0
        self.current_time = 0
        self.insulin_lockout_duration = 15  # minutes
        
    def _calculate_tdi(self):
        """Calculate Total Daily Insulin (TDI) based on patient weight."""
        return self.patient_weight * 0.55
    
    def _calculate_carb_ratio(self):
        """Calculate carbohydrate-to-insulin ratio."""
        return 500 / self.tdi
    
    def _calculate_isf(self):
        """Calculate Insulin Sensitivity Factor (ISF)."""
        return 1500 / self.tdi
    
    def set_current_time(self, time_minutes):
        """Update current simulation time."""
        self.current_time = time_minutes
    
    def is_insulin_locked_out(self):
        """Check if insulin delivery is in lockout period."""
        return (self.current_time - self.last_insulin_time) < self.insulin_lockout_duration
    
    def calculate_meal_bolus(self, carbs_grams):
        """
        Calculate meal bolus dose.
        
        Args:
            carbs_grams (float): Amount of carbohydrates in grams
            
        Returns:
            float: Bolus insulin dose in units
        """
        if carbs_grams <= 0:
            return 0.0
            
        bolus_dose = carbs_grams / self.carb_ratio
        return bolus_dose
    
    def calculate_correction_dose(self, current_glucose_mgdl, target_glucose_mgdl=120):
        """
        Calculate correction dose based on current and target glucose.
        
        Args:
            current_glucose_mgdl (float): Current blood glucose in mg/dL
            target_glucose_mgdl (float): Target blood glucose in mg/dL
            
        Returns:
            float: Correction insulin dose in units (can be negative)
        """
        glucose_difference = current_glucose_mgdl - target_glucose_mgdl
        correction_dose = glucose_difference / self.isf
        return correction_dose
    
    def deliver_bolus(self, carbs_grams, current_glucose_mgdl, target_glucose_mgdl=120):
        """
        Calculate and deliver meal bolus with correction.
        
        Args:
            carbs_grams (float): Amount of carbohydrates in grams
            current_glucose_mgdl (float): Current blood glucose in mg/dL
            target_glucose_mgdl (float): Target blood glucose in mg/dL
            
        Returns:
            dict: Contains bolus info and total insulin delivered
        """
        if self.is_insulin_locked_out():
            return {
                'bolus_dose': 0.0,
                'correction_dose': 0.0,
                'total_dose': 0.0,
                'delivered': False,
                'reason': 'Insulin lockout active'
            }
        
        # Calculate doses
        bolus_dose = self.calculate_meal_bolus(carbs_grams)
        correction_dose = self.calculate_correction_dose(current_glucose_mgdl, target_glucose_mgdl)
        total_dose = bolus_dose + correction_dose
        
        # Ensure non-negative total dose
        total_dose = max(0.0, total_dose)
        
        # Update last insulin time
        self.last_insulin_time = self.current_time
        
        return {
            'bolus_dose': bolus_dose,
            'correction_dose': correction_dose,
            'total_dose': total_dose,
            'delivered': True,
            'reason': 'Bolus delivered successfully',
            'carb_ratio': self.carb_ratio,
            'isf': self.isf
        }
    
    def deliver_correction(self, current_glucose_mgdl, target_glucose_mgdl=120):
        """
        Calculate and deliver correction dose only.
        
        Args:
            current_glucose_mgdl (float): Current blood glucose in mg/dL
            target_glucose_mgdl (float): Target blood glucose in mg/dL
            
        Returns:
            dict: Contains correction info and insulin delivered
        """
        if self.is_insulin_locked_out():
            return {
                'correction_dose': 0.0,
                'delivered': False,
                'reason': 'Insulin lockout active'
            }
        
        correction_dose = self.calculate_correction_dose(current_glucose_mgdl, target_glucose_mgdl)
        
        # Only deliver if correction is needed (positive dose)
        if correction_dose <= 0:
            return {
                'correction_dose': 0.0,
                'delivered': False,
                'reason': 'No correction needed'
            }
        
        # Update last insulin time
        self.last_insulin_time = self.current_time
        
        return {
            'correction_dose': correction_dose,
            'delivered': True,
            'reason': 'Correction delivered successfully',
            'isf': self.isf
        }
    
    def get_insulin_parameters(self):
        """Return current insulin calculation parameters."""
        return {
            'patient_weight': self.patient_weight,
            'tdi': self.tdi,
            'carb_ratio': self.carb_ratio,
            'isf': self.isf,
            'lockout_duration': self.insulin_lockout_duration
        }
    
    def update_patient_weight(self, new_weight_kg):
        """
        Update patient weight and recalculate insulin parameters.
        
        Args:
            new_weight_kg (float): New patient weight in kilograms
        """
        self.patient_weight = new_weight_kg
        self.tdi = self._calculate_tdi()
        self.carb_ratio = self._calculate_carb_ratio()
        self.isf = self._calculate_isf()

def test_insulin_calculator():
    """Test function for the insulin calculator."""
    print("=== INSULIN CALCULATOR TEST ===")
    
    # Initialize calculator for 75kg patient
    calc = InsulinCalculator(patient_weight_kg=75)
    
    print(f"Patient Parameters:")
    params = calc.get_insulin_parameters()
    for key, value in params.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n--- Test Meal Bolus ---")
    # Test meal bolus for 60g carbs at 180 mg/dL glucose
    calc.set_current_time(480)  # 8 AM
    bolus_result = calc.deliver_bolus(carbs_grams=60, current_glucose_mgdl=180)
    
    print(f"Meal: 60g carbs, BG: 180 mg/dL")
    print(f"  Bolus dose: {bolus_result['bolus_dose']:.2f} U")
    print(f"  Correction dose: {bolus_result['correction_dose']:.2f} U")
    print(f"  Total dose: {bolus_result['total_dose']:.2f} U")
    print(f"  Delivered: {bolus_result['delivered']}")
    
    print("\n--- Test Lockout ---")
    # Try to deliver another dose immediately (should be locked out)
    calc.set_current_time(485)  # 5 minutes later
    lockout_result = calc.deliver_correction(current_glucose_mgdl=200)
    print(f"Correction attempt 5 min later:")
    print(f"  Delivered: {lockout_result['delivered']}")
    print(f"  Reason: {lockout_result['reason']}")
    
    print("\n--- Test After Lockout ---")
    # Try again after lockout period
    calc.set_current_time(500)  # 20 minutes later
    correction_result = calc.deliver_correction(current_glucose_mgdl=200)
    print(f"Correction attempt 20 min later:")
    print(f"  Correction dose: {correction_result['correction_dose']:.2f} U")
    print(f"  Delivered: {correction_result['delivered']}")

if __name__ == "__main__":
    test_insulin_calculator() 