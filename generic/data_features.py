from enum import Enum

class FeatureType(Enum):
    NUMBER = 0,
    CATEGORY = 1,

korea_features = {
    # 'YEAR': ,
    # 'IDV_ID': ,
    'AREA_CODE': FeatureType.CATEGORY,
    'SEX': FeatureType.CATEGORY,
    'AGE_GROUP': FeatureType.CATEGORY,
    'HEIGHT': FeatureType.NUMBER,
    'WEIGHT': FeatureType.NUMBER,
    'WAIST': FeatureType.NUMBER,
    'SIGHT_LEFT': FeatureType.NUMBER,
    'SIGHT_RIGHT': FeatureType.NUMBER,
    'HEAR_LEFT': FeatureType.NUMBER,
    'HEAR_RIGHT': FeatureType.NUMBER,
    'BP_HIGH': FeatureType.NUMBER,
    'BP_LWST': FeatureType.NUMBER,
    'BLDS': FeatureType.NUMBER,
    'TOT_CHOLE': FeatureType.NUMBER,
    'TRIGLYCERIDE': FeatureType.NUMBER,
    'HDL_CHOLE': FeatureType.NUMBER,
    'LDL_CHOLE': FeatureType.NUMBER,
    'HMG': FeatureType.NUMBER,
    'OLIG_PROTE_CD': FeatureType.CATEGORY,
    'CREATININE': FeatureType.NUMBER,
    'SGOT_AST': FeatureType.NUMBER,
    'SGPT_ALT': FeatureType.NUMBER,
    'GAMMA_GTP': FeatureType.NUMBER,
    'SMK_STAT': FeatureType.CATEGORY,
    'DRK_YN': FeatureType.CATEGORY,
    'HCHK_CE_IN': FeatureType.CATEGORY,
    'CRS_YN': FeatureType.CATEGORY,
    'TTR_YN': FeatureType.CATEGORY,
    'TTH_MSS_YN': FeatureType.CATEGORY,
    'ODT_TRB_YN': FeatureType.CATEGORY,
    'WSDM_DIS_YN': FeatureType.CATEGORY,


    # 'DATE':,
}
korea_dental_features = ['HCHK_CE_IN','CRS_YN','TTH_MSS_YN','ODT_TRB_YN','WSDM_DIS_YN','TTR_YN']
korea_features_to_remove = ['DATE', 'YEAR', 'IDV_ID']

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


