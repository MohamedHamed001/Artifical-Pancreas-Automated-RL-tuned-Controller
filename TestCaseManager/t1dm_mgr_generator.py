#!/usr/bin/env python3

import math
import random
import os

# Helper function to format scientific notation with a fixed 3-digit exponent
def format_scientific_notation(value_float, total_width, precision):
    """Formats a float in scientific notation with a fixed 3-digit exponent width."""
    # Handle NaN, Infinity explicitly if necessary
    if math.isnan(value_float):
         return f"{float('nan'):>{total_width}}".replace('NAN', ' NaN') # Right align NaN
    if math.isinf(value_float):
         return f"{float('inf'):>{total_width}}".replace('INF', ' Inf') # Right align Inf

    if value_float == 0.0:
        # For 0.0, format to the desired precision and manually set exponent to +000
        # Use .f format first to control precision before 'e'
        mantissa_part = f"{0.0:.{precision}f}"
        formatted_value = f"{mantissa_part}e+000"
    else:
        # Format the non-zero value using standard scientific notation first
        formatted_value = f"{value_float:.{precision}e}"

        # Split into mantissa and exponent
        parts = formatted_value.split('e')
        mantissa = parts[0]
        exponent_sign = parts[1][0] # + or -
        # Get the absolute value of the exponent and format with 3 digits
        exponent_value = int(parts[1][1:])
        formatted_exponent = f"{exponent_sign}{abs(exponent_value):03d}"

        # Combine mantissa and formatted exponent
        formatted_value = f"{mantissa}e{formatted_exponent}"

    # Pad with spaces to reach total_width (right alignment is typical for numbers)
    return formatted_value.rjust(total_width)


print("======Welcome to T1DM Patient Virtual System Test Case Generator======")

# Ask for generation mode
while True:
    try:
        mode = input("Choose generation mode (m/a):\nm - Manual input\na - Automatic generation\nEnter choice: ").lower()
        if mode in ['m', 'a']:
            break
        else:
            print("Error: Please enter either 'm' for manual or 'a' for automatic generation.")
    except ValueError:
        print("Error: Please enter a valid mode.")

# Ask for simulation time
while True:
    try:
        simulation_time = int(input("Enter simulation time in minutes [>0]: "))
        if simulation_time > 0:
            break
        else:
            print("Error: Simulation time must be greater than 0 minutes.")
    except ValueError:
        print("Error: Please enter a valid number.")

numCases = int(input("Enter the number of cases you will generate: "))
currentCase = 1

output_dir = "TestData"
os.makedirs(output_dir, exist_ok=True)

print("NOTE: Enter real values within range only!")

with open("TestCases.txt", "w") as myfile:
    myfile.write("")

while currentCase <= numCases:
    print(f"\nTest Case [{currentCase}]")

    if mode == 'm':
        # Manual input mode
        while True:
            try:
                bw = float(input("Body Weight (kg) [2-200]: "))
                if 2 <= bw <= 200:
                    break
                else:
                    print("Error: Body Weight must be between 2 and 200 kg.")
            except ValueError:
                print("Error: Please enter a valid number.")

        while True:
            try:
                meal_count = int(input("Number of Meals [0-10]: "))
                if 0 <= meal_count <= 10:
                    break
                else:
                    print("Error: Number of Meals must be between 0 and 10.")
            except ValueError:
                print("Error: Please enter a valid integer.")

        meals = []
        for meal_num in range(1, meal_count + 1):
            print(f"Meal {meal_num}:")

            while True:
                try:
                    meal_time_minutes = int(input("    Meal Time (minutes from start) [>=0]: "))
                    if meal_time_minutes >= 0 and meal_time_minutes + 40 <= simulation_time:
                        meals.append({'meal_time_minutes': meal_time_minutes})
                        break
                    else:
                         print(f"Error: Meal Time must be between 0 and {simulation_time-40} minutes.")
                except ValueError:
                    print("Error: Please enter a valid number.")

            while True:
                try:
                    carb_amount = float(input("    Carb Amount (grams) [0-200]: "))
                    if 0 <= carb_amount <= 200:
                        meals[-1]['carb_amount'] = carb_amount
                        break
                    else:
                        print("Error: Carb Amount must be between 0 and 200 grams.")
                except ValueError:
                    print("Error: Please enter a valid number.")

        # Exercise input for manual mode
        while True:
            try:
                exercise_count = int(input("Number of Exercise Sessions [0-5]: "))
                if 0 <= exercise_count <= 5:
                    break
                else:
                    print("Error: Number of Exercise Sessions must be between 0 and 5.")
            except ValueError:
                print("Error: Please enter a valid integer.")

        exercises = []
        for ex_num in range(1, exercise_count + 1):
            print(f"Exercise Session {ex_num}:")
            
            while True:
                try:
                    ex_start_time = int(input("    Start Time (minutes from start) [>=0]: "))
                    if ex_start_time >= 0 and ex_start_time < simulation_time:
                        exercises.append({'start_time': ex_start_time})
                        break
                    else:
                        print(f"Error: Start Time must be between 0 and {simulation_time-1} minutes.")
                except ValueError:
                    print("Error: Please enter a valid number.")

            while True:
                try:
                    ex_duration = int(input("    Duration (minutes) [>0]: "))
                    if ex_duration > 0 and ex_start_time + ex_duration <= simulation_time:
                        exercises[-1]['duration'] = ex_duration
                        break
                    else:
                        print(f"Error: Duration must be between 1 and {simulation_time - ex_start_time} minutes.")
                except ValueError:
                    print("Error: Please enter a valid number.")
    else:
        # Automatic generation mode
        # Generate body weight between 2 and 200 kg
        bw = round(random.uniform(2, 200), 2)
        print(f"Generated Body Weight: {bw} kg")

        # Generate number of meals between 0 and 10
        meal_count = random.randint(0, 10)
        print(f"Generated Number of Meals: {meal_count}")

        meals = []
        # Generate meals with increasing time intervals
        current_time = 0
        for meal_num in range(1, meal_count + 1):
            # Add random time interval between 60 and 240 minutes
            time_interval = random.randint(60, 240)
            meal_time_minutes = current_time + time_interval
            
            # Ensure meal and its 40-minute effect fit within simulation time
            if meal_time_minutes + 40 > simulation_time:
                break
                
            current_time = meal_time_minutes

            # Generate carb amount between 0 and 200 grams
            carb_amount = round(random.uniform(0, 200), 2)

            meals.append({
                'meal_time_minutes': meal_time_minutes,
                'carb_amount': carb_amount
            })
            print(f"Generated Meal {meal_num}: Time = {meal_time_minutes} minutes, Carbs = {carb_amount:.2f} grams")

        # Exercise generation for automatic mode
        exercise_count = random.randint(0, 5)
        print(f"Generated Number of Exercise Sessions: {exercise_count}")

        exercises = []
        current_time = 0
        for ex_num in range(1, exercise_count + 1):
            # Add random time interval between 30 and 120 minutes
            time_interval = random.randint(30, 120)
            ex_start_time = current_time + time_interval
            
            # Generate duration between 15 and 60 minutes
            ex_duration = random.randint(15, 60)
            
            # Ensure exercise session fits within simulation time
            if ex_start_time + ex_duration > simulation_time:
                break
                
            current_time = ex_start_time + ex_duration

            exercises.append({
                'start_time': ex_start_time,
                'duration': ex_duration
            })
            print(f"Generated Exercise Session {ex_num}: Start = {ex_start_time} minutes, Duration = {ex_duration} minutes")

    outputFormat = f"Test Case [{currentCase}]\n" \
                   f"Body Weight: {bw} kg\n" \
                   f"Simulation Time: {simulation_time} minutes\n\n" \
                   f"Meal Count: {meal_count}"

    with open("TestCases.txt", "a") as myfile:
        print(outputFormat, file=myfile)
        for idx, meal in enumerate(meals, start=1):
            print(f"Meal {idx} Time: {meal['meal_time_minutes']} minutes, Carb Amount: {meal['carb_amount']} grams", file=myfile)
        print("\nExercise Sessions:", file=myfile)
        for idx, exercise in enumerate(exercises, start=1):
            print(f"Exercise {idx}: Start = {exercise['start_time']} minutes, Duration = {exercise['duration']} minutes", file=myfile)
        print("\n", file=myfile)

    # Generate meal data
    meal_data_filename = os.path.join(output_dir, f"MealData_case{currentCase}.data")
    exercise_data_filename = os.path.join(output_dir, f"ExerciseData_case{currentCase}.data")

    # Sort meals by meal_time_minutes
    sorted_meals_list = sorted(meals, key=lambda x: x['meal_time_minutes'])

    # Create list for total amount (mmol) per minute from 0 up to simulation_time
    amount_per_minute_mmol = [0.0] * (simulation_time + 1)

    # Populate meal effects with 40-minute duration
    for meal in sorted_meals_list:
        meal_start_minute = meal['meal_time_minutes']
        mmol_amount = meal['carb_amount'] * 1000.0 / 180.1577  # Convert grams to mmol

        # Only process meals with positive carb amounts
        if meal_start_minute >= 0 and mmol_amount > 1e-9:
            effect_end_minute = meal_start_minute + 40

            # Add amount to minutes within the 40-minute effect window
            for m in range(meal_start_minute, effect_end_minute + 1):
                if 0 <= m <= simulation_time:
                    amount_per_minute_mmol[m] += mmol_amount

    # Collect all required output minutes for meals
    required_output_minutes = set()
    required_output_minutes.add(0)  # Start time
    required_output_minutes.add(simulation_time)  # End time

    # Add points for each meal window
    for meal in sorted_meals_list:
        start = meal['meal_time_minutes']
        end = start + 40
        
        # Add points before, at start, during, and after the meal
        required_output_minutes.add(start - 1)  # Point before meal
        required_output_minutes.add(start)      # Start of meal
        required_output_minutes.add(end)        # End of meal
        required_output_minutes.add(end + 1)    # Point after meal

    # Ensure all required points are within the simulation bounds and non-negative
    required_output_minutes = {m for m in required_output_minutes if m >= 0 and m <= simulation_time}

    # Sort the required output minutes
    sorted_output_minutes = sorted(list(required_output_minutes))

    # Create the final output data points for meals
    output_data_points = []
    for minute in sorted_output_minutes:
        amount_mmol = amount_per_minute_mmol[minute] if minute >= 0 and minute <= simulation_time else 0.0
        amount_mmol_per_kg = amount_mmol / bw if bw > 0 else 0.0
        output_data_points.append((float(minute), amount_mmol_per_kg))

    with open(meal_data_filename, "w") as myfile:
        myfile.write("# Table format: 1D\n")
        for time_minutes_float, amount_mmol_per_kg in output_data_points:
            time_str = format_scientific_notation(time_minutes_float, 25, 14)
            amount_str = format_scientific_notation(amount_mmol_per_kg, 25, 14)
            myfile.write(f"{time_str}          {amount_str}\n")

    # Generate exercise data
    # Create list for exercise status (0 or 1) per minute
    exercise_status = [0] * (simulation_time + 1)

    # Populate exercise status
    for exercise in exercises:
        start = exercise['start_time']
        duration = exercise['duration']
        end = start + duration

        # Set status to 1 during exercise
        for m in range(start, end + 1):
            if 0 <= m <= simulation_time:
                exercise_status[m] = 1

    # Collect required output minutes for exercise
    exercise_output_minutes = set()
    exercise_output_minutes.add(0)  # Start time
    exercise_output_minutes.add(simulation_time)  # End time

    # Add points for each exercise session
    for exercise in exercises:
        start = exercise['start_time']
        end = start + exercise['duration']
        
        # Add points before, at start, during, and after exercise
        exercise_output_minutes.add(start - 1)  # Point before exercise
        exercise_output_minutes.add(start)      # Start of exercise
        exercise_output_minutes.add(end)        # End of exercise
        exercise_output_minutes.add(end + 1)    # Point after exercise

    # Ensure all required points are within the simulation bounds and non-negative
    exercise_output_minutes = {m for m in exercise_output_minutes if m >= 0 and m <= simulation_time}

    # Sort the required output minutes
    sorted_exercise_minutes = sorted(list(exercise_output_minutes))

    # Create the final output data points for exercise
    exercise_data_points = []
    for minute in sorted_exercise_minutes:
        status = exercise_status[minute]
        exercise_data_points.append((float(minute), float(status)))

    with open(exercise_data_filename, "w") as myfile:
        myfile.write("# Table format: 1D\n")
        for time_minutes_float, status in exercise_data_points:
            time_str = format_scientific_notation(time_minutes_float, 25, 14)
            status_str = format_scientific_notation(status, 25, 14)
            myfile.write(f"{time_str}          {status_str}\n")

    currentCase += 1

print("\nWe have generated the test cases in TestCases.txt file.")
print("We have generated the meal data files for each test case.")
print("We have generated the exercise data files for each test case.")