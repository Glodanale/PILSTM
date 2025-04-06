import pandas as pd

df = pd.read_csv("4_ReportLoss.csv")

filtered_df = df[df['MSEc_Percentile'] > 0.95]

sequence_numbers = filtered_df['Sequence_Number'].tolist()

with open("largeLoss.txt", "w") as file:
        for item in sequence_numbers:
            file.write(f"{item}, ")

#after running this go to the txt file and delete the comma after the final group name