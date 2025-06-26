import pandas as pd
import numpy as np
import os

class MealParser:
    """
    Parses meal and exercise data from test case files for diabetes simulation.
    """
    
    def __init__(self):
        self.meal_data = None
        self.exercise_data = None
        
    def parse_test_case(self, test_case_file):
        """
        Parses a test case file to extract meal and exercise information.
        
        Args:
            test_case_file (str): Path to the test case file
            
        Returns:
            tuple: (meal_dataframe, exercise_dataframe)
        """
        meals = []
        exercises = []
        
        try:
            with open(test_case_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                
                # Parse meal data - handle both formats
                if line.startswith("Meal Time:") or ("Time:" in line and "Carb" in line and "Meal" in line):
                    try:
                        # Extract time and carbs from lines like:
                        # "Meal Time: 480 min, Carbs: 80 g" or
                        # "Meal 1 Time: 8 minutes, Carb Amount: 40.0 grams"
                        parts = line.split(',')
                        
                        # Extract time part more carefully
                        time_section = parts[0].split(':')[1].strip()
                        # Extract only the number from the time section
                        import re
                        time_match = re.search(r'(\d+\.?\d*)', time_section)
                        if time_match:
                            time_value = float(time_match.group(1))
                        else:
                            continue
                        
                        # Extract carbs part more carefully  
                        carbs_section = parts[1].split(':')[1].strip()
                        carbs_match = re.search(r'(\d+\.?\d*)', carbs_section)
                        if carbs_match:
                            carbs_value = float(carbs_match.group(1))
                        else:
                            continue
                        
                        meals.append({
                            'time': int(time_value),
                            'carbs': carbs_value
                        })
                    except (IndexError, ValueError) as e:
                        print(f"Warning: Could not parse meal line: {line}")
                        continue
                
                # Parse exercise data  
                elif line.startswith("Exercise"):
                    # Extract exercise info from line like "Exercise 1: Start = 300 minutes, Duration = 60 minutes"
                    if "Start" in line and "Duration" in line:
                        parts = line.split(',')
                        start_part = parts[0].split('=')[1].strip().replace(' minutes', '')
                        duration_part = parts[1].split('=')[1].strip().replace(' minutes', '')
                        
                        start_time = int(start_part)
                        duration = int(duration_part)
                        
                        # Create exercise start and end events
                        exercises.append({'time': start_time, 'active': 1})
                        exercises.append({'time': start_time + duration, 'active': 0})
                        
        except FileNotFoundError:
            print(f"Error: Test case file {test_case_file} not found")
            
        # Convert to DataFrames
        meal_df = pd.DataFrame(meals) if meals else pd.DataFrame(columns=['time', 'carbs'])
        exercise_df = pd.DataFrame(exercises) if exercises else pd.DataFrame(columns=['time', 'active'])
        
        # Sort by time
        if not meal_df.empty:
            meal_df = meal_df.sort_values('time').reset_index(drop=True)
        if not exercise_df.empty:
            exercise_df = exercise_df.sort_values('time').reset_index(drop=True)
            
        self.meal_data = meal_df
        self.exercise_data = exercise_df
        
        return meal_df, exercise_df
    
    def save_to_data_files(self, meal_file_path, exercise_file_path):
        """
        Saves parsed meal and exercise data to data files compatible with Hovorka model.
        
        Args:
            meal_file_path (str): Path to save meal data
            exercise_file_path (str): Path to save exercise data
        """
        if self.meal_data is not None:
            # Create meal data file with start/end format expected by Hovorka model
            meal_data_expanded = []
            
            for _, row in self.meal_data.iterrows():
                # Meal start (non-zero carbs)
                meal_data_expanded.append({'time': row['time'], 'carbs': row['carbs']})
                # Meal end (zero carbs) - 1 minute later to indicate end
                meal_data_expanded.append({'time': row['time'] + 1, 'carbs': 0.0})
            
            if meal_data_expanded:
                meal_df_expanded = pd.DataFrame(meal_data_expanded)
                meal_df_expanded = meal_df_expanded.sort_values('time')
            else:
                meal_df_expanded = pd.DataFrame(columns=['time', 'carbs'])
            
            # Save in space-separated format
            with open(meal_file_path, 'w') as f:
                f.write("# Time(min) Carbs(g)\n")
                for _, row in meal_df_expanded.iterrows():
                    f.write(f"{row['time']} {row['carbs']}\n")
                    
            print(f"Meal data saved to {meal_file_path}")
            
        if self.exercise_data is not None:
            # Save exercise data
            with open(exercise_file_path, 'w') as f:
                f.write("# Time(min) Active(0/1)\n")
                for _, row in self.exercise_data.iterrows():
                    f.write(f"{row['time']} {row['active']}\n")
                    
            print(f"Exercise data saved to {exercise_file_path}")
    
    def get_meal_summary(self):
        """Returns a summary of parsed meal data."""
        if self.meal_data is None or self.meal_data.empty:
            return "No meals found"
            
        summary = f"Found {len(self.meal_data)} meals:\n"
        for _, row in self.meal_data.iterrows():
            summary += f"  - Time: {row['time']} min, Carbs: {row['carbs']} g\n"
        return summary
    
    def get_exercise_summary(self):
        """Returns a summary of parsed exercise data."""
        if self.exercise_data is None or self.exercise_data.empty:
            return "No exercise sessions found"
            
        # Group exercise events into sessions
        sessions = []
        current_session = None
        
        for _, row in self.exercise_data.iterrows():
            if row['active'] == 1:  # Exercise start
                current_session = {'start': row['time']}
            elif row['active'] == 0 and current_session:  # Exercise end
                current_session['end'] = row['time']
                current_session['duration'] = current_session['end'] - current_session['start']
                sessions.append(current_session)
                current_session = None
        
        summary = f"Found {len(sessions)} exercise sessions:\n"
        for i, session in enumerate(sessions, 1):
            summary += f"  - Session {i}: Start: {session['start']} min, Duration: {session['duration']} min\n"
        return summary

def test_parser():
    """Test function for the meal parser."""
    parser = MealParser()
    
    # Test with the provided test case
    test_case_path = "../../TestData/TestCases.txt"
    if os.path.exists(test_case_path):
        meal_df, exercise_df = parser.parse_test_case(test_case_path)
        
        print("=== MEAL PARSER TEST ===")
        print(parser.get_meal_summary())
        print(parser.get_exercise_summary())
        
        # Save to data files
        os.makedirs("../data/test_cases", exist_ok=True)
        parser.save_to_data_files(
            "../data/test_cases/MealData_test.data",
            "../data/test_cases/ExerciseData_test.data"
        )
    else:
        print(f"Test case file not found: {test_case_path}")

if __name__ == "__main__":
    test_parser() 