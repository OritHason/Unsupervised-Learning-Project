from enum import Enum


class FeatureType(Enum):
    NUMBER = 0,
    CATEGORY = 1,


work_features = {
    'Employee_ID': FeatureType.NUMBER,
    'Age': FeatureType.NUMBER,
    'Department': FeatureType.CATEGORY,
    'Job_Level': FeatureType.CATEGORY,
    'Years_at_Company': FeatureType.NUMBER,
    'Monthly_Hours_Worked': FeatureType.NUMBER,
    'Remote_Work': FeatureType.CATEGORY,
    'Meetings_per_Week': FeatureType.NUMBER,
    'Tasks_Completed_Per_Day': FeatureType.NUMBER,
    'Overtime_Hours_Per_Week': FeatureType.NUMBER,
    'Work_Life_Balance': FeatureType.CATEGORY,
    'Job_Satisfaction': FeatureType.NUMBER,
    'Productivity_Score': FeatureType.NUMBER,
    'Annual_Salary': FeatureType.NUMBER,
    'Absences_Per_Year': FeatureType.NUMBER
}
work_features_to_remove = ['Employee_ID']


