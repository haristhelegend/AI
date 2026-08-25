import pandas as pd

# Raw Data: Attendance in various formats
data = {
    'Attendance_Status': ['Present', 'Absent', 'Present', 'Present', 'Absent']
}
df = pd.DataFrame(data)

# The model cannot understand words like 'Present' or 'Absent'.
# We must convert them into numbers (0 and 1). This is called "Label Encoding."
df['Attendance_Numeric'] = df['Attendance_Status'].map({'Present': 1, 'Absent': 0})

print("Before Engineering:")
print(df['Attendance_Status'])
print("\nAfter Engineering (Ready for the Model):")
print(df[['Attendance_Status', 'Attendance_Numeric']])