# Matthew_Lambalot_Data_Science_Capstone_Project
Fall 2025 Data Science Capstone Project

# App usage instructions 

This is a pretty simple app designed to take readily available data from the user and give them an estimated risk percentage of getting a coronary heart disease in the next 10 years. 

To do this the user simply inputs their age, sex, height, and weight, as well as if they smoke or have diabetes. 
Then clicking the "Estimate 10-Year CVD Risk button at the bottom" will output your risk score as well as some information on how to reduce it and what it means. 

## What does the information mean and what are each of the graphs suppose to show? 

### Risk percentage and factor contributions 
The first section gives you your CVD Risk percentage, which is the chance you get a CHD within the next 10 years. 

It then explains what that means, and gives you the biggest factor that you can change to reduce this risk. 

The factor contribution table shows what percent each of your factors plays into your total CVD risk to give a better idea of how they affect each other 

### Risk percentage histogram 
This graph is simply used to show the how your risk factor compares to those within your age range (+/- 2 years)

### Risk vs age graph 
This graph is another way to visualize how your CVD risk compares to the entire cohort to better contextualize what is normal 

### Lifestyle change impact on CHD bar graph 
This is a simple bar graph to show how changing your lifetyle can decrease your 10 year CHD risk

### 10 year risk trajectory 
This is the final graph and is used to illustrate how your CHD risk will increase over the next 10 years, and compares that to how your risk factor would change if you made the apprropriate lifestyle changes 

# Model explanation 

## Dataset selection 
The framingham dataset was selected as it is a longitudinal study that takes in many risk factors, then shows whether that patient developed a CHD within a 10 year period. 

This is an ideal dataset for a model trying to use those risk factors as predictors for relative CHD risk over 10 years. 

The external dataset used to test the model on had all the same factors as the framingham test;however, it didn't track whether they got CHD but instead assigned a high medium low risk associated with their chances of developing one. 

This dataset was selected as no free external datasets could be found that was a similar study type as the framingham study. To better use the data, those with high risk levels were assigned to have gotten a CHD within the 10 year period, which likely over estimates the total patients who developed one. 

## Data interpretation 
Test AUC= .683
External AUC= .563 

The test AUC is on the weak side but somewhat acceptable. This is likely due to the many factors not being included within the model to ensure ease of use. I figured people would not have easy access to all of their bloodwork and other health numbers. 

The external AUC of .563 is weak and only slightly above random chance. This is likely due to the two datasets measuring different things. One assigns relative risk while the other tracks whether or not the patient actually got a CHD. Better validation would likley use more similar datasets; however, due to access of data this was not possible. 



