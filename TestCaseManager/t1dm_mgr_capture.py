import re
import os
import matplotlib.pyplot as plt
import natsort

def plot_test_case(test_case_name, test_case_num):
    with open('testReports/' + test_case_name) as f:
        lines = f.readlines()
    t, glucose_levels, insulin_levels, carb_intake = [], [], [], []
    for line in lines:
        line = line.strip()
        # Assuming the line format is: Time: t, Glucose Level: g, Insulin Level: i, Carb Intake: c
        match = re.match(r'Time: ([\d\.]+), Glucose Level: ([\d\.]+), Insulin Level: ([\d\.]+), Carb Intake: ([\d\.]+)', line)
        if match:
            t.append(float(match.group(1)))
            glucose_levels.append(float(match.group(2)))
            insulin_levels.append(float(match.group(3)))
            carb_intake.append(float(match.group(4)))
    # Assuming the test case parameters are stored at the beginning of the file
    with open('testReports/' + test_case_name) as f:
        first_line = f.readline().strip()
    title = f'Test Case {test_case_num}\n{first_line}'
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle(title)
    axs[0].plot(t, glucose_levels, 'tab:red')
    axs[0].set_ylabel('Glucose Level (mg/dL)')
    axs[0].set_title('Glucose Level Over Time')
    axs[1].plot(t, insulin_levels, 'tab:blue')
    axs[1].set_ylabel('Insulin Level (units)')
    axs[1].set_title('Insulin Level Over Time')
    axs[2].bar(t, carb_intake, width=5, color='tab:green')
    axs[2].set_ylabel('Carb Intake (grams)')
    axs[2].set_title('Carb Intake Over Time')
    axs[2].set_xlabel('Time (minutes)')
    plt.tight_layout()
    plt.show()

for file_num, file_name in enumerate(natsort.natsorted(os.listdir('testReports'))):
    plot_test_case(file_name, file_num + 1)
