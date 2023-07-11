from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import ast


df = pd.read_csv('msc_collated_data_20230709.csv')

# Selecting columns of interest
df_roi = df[['session_number', 
             'session_type', 
             'participant_number', 
             'key_resp.rt', 
             'task', 
             'valid_cue', 
             'cue_dir', 
             'correct_response',
             'mean_aai']]

# Rename RT column
df_roi.rename(columns={'key_resp.rt':'rt'}, inplace=True)

# Drop nans
df_roi = df_roi.dropna()

# Get first reaction time to usable number
df_roi['rt'] = df_roi['rt'].apply(ast.literal_eval)
df_roi['rt'] = df_roi['rt'].apply(lambda x: x[0])

# Keep only posner tasks
df_roi = df_roi.drop(df_roi[df_roi['task'] == 'baseline0'].index)
df_roi = df_roi.drop(df_roi[df_roi['task'] == 'baseline1'].index)
df_roi = df_roi.drop(df_roi[df_roi['task'] == 'neurofeedback0'].index)
df_roi = df_roi.drop(df_roi[df_roi['task'] == 'neurofeedback1'].index)
df_roi = df_roi.drop(df_roi[df_roi['task'] == 'neurofeedback2'].index)
df_roi = df_roi.drop(df_roi[df_roi['task'] == 'neurofeedback3'].index)
assert df_roi.task.unique()[0]=='posner0'
assert df_roi.task.unique()[1]=='posner1'
assert len(df_roi.task.unique())==2

# Drop invalid trials
df_roi = df_roi.drop(df_roi[df_roi['valid_cue'] == False].index)
assert df_roi.valid_cue.unique()==True

# Keep only left cues
df_roi = df_roi.drop(df_roi[df_roi['cue_dir'] == 'centre'].index)
df_roi = df_roi.drop(df_roi[df_roi['cue_dir'] == 'right'].index)
assert df_roi.cue_dir.unique()=='left'

# Keep only correct responses:
df_roi = df_roi.drop(df_roi[df_roi['correct_response'] == False].index)
assert df_roi.correct_response.unique()==True

# Just look at session 1 posner a versus session 2 posner b
df_roi = df_roi.drop(df_roi[(df_roi['session_number'] == 'ses-01') & (df_roi['task'] == 'posner1') |
                (df_roi['session_number'] == 'ses-02') & (df_roi['task'] == 'posner0')].index)
# Double check
assert all(df_roi.loc[df_roi['session_number'] == 'ses-01', 'task'] == 'posner0')
assert all(df_roi.loc[df_roi['session_number'] == 'ses-02', 'task'] == 'posner1')

# Function to determine the group of each participant
def set_group(x):
    if x.nunique() == 2:
        return 'sham'
    else:
        return 'active'

# Apply the function to each participant
df_roi['group'] = df_roi.groupby('participant_number')['session_type'].transform(set_group)
df_roi.group.count()

# Rename session_number values for plotting
df_roi['session_number'] = df_roi['session_number'].replace({'ses-01': 'pos1a', 'ses-02': 'pos2b'})

# Function to determine the group of each participant
def set_group(x):
    if x.nunique() == 2:
        return 'sham'
    else:
        return 'active'

# Apply the function to each participant
df_roi['group'] = df_roi.groupby('participant_number')['session_type'].transform(set_group)
df_roi.group.count()

# Rename session_number values for plotting
df_roi['session_number'] = df_roi['session_number'].replace({'ses-01': 'pos1a', 'ses-02': 'pos2b'})

# Interaction plot for session, condition and RT
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
interaction_plot = sns.pointplot(x='session_number', y='rt', hue='group', data=df_roi, dodge=True, markers=['o', 's'], capsize=.1, errwidth=1, palette='colorblind')

plt.title('Session number, condition and reaction time')
plt.xlabel('session / posner task')
plt.ylabel('Key Response Time')
plt.legend(title='Condition')
plt.show()

# Define model
# 'C()' indicates that we want to treat the variables as categorical
pd.set_option('display.float_format', '{:.3f}'.format)
formula_RT = 'rt ~ C(group) + C(session_number) + C(group):C(session_number)'
model = ols(formula_RT, data=df_roi).fit()

# Create anova table with stats
anova_table = sm.stats.anova_lm(model, typ=2)
print('**** RT anova model ****')
print('Formula: ', formula_RT, '\n')
print(anova_table)
pd.reset_option('display.float_format')

# Plot for AAI effects
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
interaction_plot = sns.pointplot(x='session_number', y='mean_aai', hue='group', data=df_roi, dodge=True, markers=['o', 's'], capsize=.1, errwidth=1, palette='colorblind')

plt.title('Session number, condition and mean AAI')
plt.xlabel('session / posner task')
plt.ylabel('Mean AAI')
plt.legend(title='Condition')
plt.show()

# Define model
# 'C()' indicates that we want to treat the variables as categorical
# pd.set_option('display.float_format', '{:.3f}'.format)
formula_AAI = 'mean_aai ~ C(group) + C(session_number) + C(group):C(session_number)'
model = ols(formula_AAI, data=df_roi).fit()

# Create anova table with stats
anova_table = sm.stats.anova_lm(model, typ=2)
print('\n\n**** mean AAI anova model ****')
print('Formula: ', formula_AAI, '\n')
print(anova_table)
# pd.reset_option('display.float_format')