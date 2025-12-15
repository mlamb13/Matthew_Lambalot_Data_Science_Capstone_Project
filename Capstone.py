## Matthew Lambalot Capstone

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# python -m streamlit run Capstone.py



## Load in model and data 
model=joblib.load("cvd_logistic_model.pkl")
CHD_df= pd.read_csv("framingham.csv")
inputs=["age", "male", "currentSmoker", "BMI", "diabetes"]  ## inputs for user to determine CHD risk. I focused on ones that people would have on hand so this calculator is mainly something to give a general idea of CHD risk and mitigation.
feature_names = model.named_steps["preprocess"].get_feature_names_out() ## input names after preprossesing. Includes the scaled numeric inputs and the binaries
feature_map = {"numeric__age": "Age", "numeric__BMI": "BMI", "binary__male": "Male sex", "binary__currentSmoker": "Current smoker", "binary__diabetes": "Diabetes"}

### Fundction definitions to be used later in the code 


## Create function to determine the percentile CHD risk someone is compared to their age group (+/- 2 years)

def CHD_percentile(age, risk, data, window=2):
    subset = data[ (data["age"] >= age - window) & (data["age"] <= age + window)]

    if len(subset) == 0:
        return np.nan

    X = subset[["age", "male", "currentSmoker", "BMI", "diabetes"]]
    risks = model.predict_proba(X)[:, 1]

    return (risks < risk).mean() * 100


## Create a function that will calcualte the risk adjustment for someone if they change their smoking status or their BMI back down to mean levels for their age. These are the only two factors under a person's control and I want to show how they can help mitigate CHD risk

def risk_ajusted_CHD(input_df, baseline_risk):
    ajusted_CHD = []        

    # Quit smoking: Essentially just subs out the values provided with one where the person doesn't smoke and reruns the model on it to see the difference in risk
    if input_df["currentSmoker"].iloc[0] == 1:
        s = input_df.copy()
        s["currentSmoker"] = 0
        risk = model.predict_proba(s)[:, 1][0]
        ajusted_CHD.append(("Quit Smoking", baseline_risk - risk))

    # Lower BMI: Same as smoking, but instead subs their BMI value with one that is an average BMI to keep it realistic
    s = input_df.copy()
    s["BMI"] = CHD_df["BMI"].mean()
    risk = model.predict_proba(s)[:, 1][0]
    ajusted_CHD.append(("Lower BMI", baseline_risk - risk))

    if not ajusted_CHD:
        return ("Risk factors in line with population", 0.0)

    return max(ajusted_CHD, key=lambda x: x[1])

## Define function that will be used to separate out risk factors for patient and show the percentage of each that contribute to their CVD risk factor 
## Decouples the impact of each risk factor on the CHD risk overall. Helps to show why someone may have higher than average CHD risk 

def risk_decomposition(input_df):
    # Extract logistic regression
    lr = model.named_steps["classifier"]
    coef = lr.coef_[0]

    # Transform input the same way the model sees it
    X_trans = model.named_steps["preprocess"].transform(input_df)

    contributions = X_trans[0] * coef

    df = pd.DataFrame({
        "feature": feature_names,
        "contribution": contributions
    })
    df["feature"] = df["feature"].map(feature_map) ## Change features to match the same titles as inputs that user enters
    df["abs_contribution"] = df["contribution"].abs()
    df["percent"] = 100 * df["abs_contribution"] / df["abs_contribution"].sum()

    return df.sort_values("percent", ascending=False)

## Define function that will be used to graph how CVD risk for patient will change of the next 10 years 
## This one is pretty simple. Just keeps all other parameters constant and iterates the model by increasing the age by 1 to show how CHD risk increases over time if lifestyle changes are not made

def age_risk_trajectory(input_df, years=10):
    rows = []

    start_age = int(input_df["age"].iloc[0])

    for a in range(start_age, start_age + years + 1):
        s = input_df.copy()
        s["age"] = a
        risk = model.predict_proba(s)[:, 1][0]
        rows.append((a, risk))

    return pd.DataFrame(rows, columns=["age", "risk"])

## This function replaces the input df with the improved lifestyle changes. It will be used to graph the 10 year difference of the improved lifestyle to show how making changes now can compound into helping your CHD risk in the future
def improved_profile(input_df):
    s = input_df.copy()

    # Quit smoking if applicable
    if s["currentSmoker"].iloc[0] == 1:
        s["currentSmoker"] = 0

    # Lower BMI to population mean
    s["BMI"] = CHD_df["BMI"].mean()

    return s



## Streamlit app 

## App UI 

st.title("Estimate Your 10-Year Risk of Coronary Heart Disease (CHD)")

st.write("Input your information below then click the button at the bottom. This tool will estiamte your 10-year risk of cardiovascular disease")

## Inputs for user. These will be used to run the model on

## Establish min and max age ranges to prevent crashes. Current dataset does not encompass any age prior to 30
age_min= int(CHD_df["age"].min())
age_max= int(CHD_df["age"].max())


age= st.number_input("Age (years)", min_value=age_min, max_value=age_max, value=40)
sex= st.selectbox("Sex", ["Female", "Male"])
male=1 if sex =="Male" else 0
smoker=st.selectbox("Do you currently smoke?", ["No", "Yes"])
smoker_status=1 if smoker =="Yes" else 0
diabetes= st.selectbox("Do you have diabetes?", ["No", "Yes"])
diabetes_status=1 if diabetes =="Yes" else 0


## BMI Calculator 
height= st.number_input("Height (cm)", min_value=120, max_value=220, value=170)
weight= st.number_input("Weight (kg)", min_value=40, max_value=200, value=75)
bmi= weight/((height/100)**2) if height>0 else np.nan


# Create model dataframe input 

input_df= pd.DataFrame([{
    "age": age,
    "male": male,
    "currentSmoker": smoker_status, 
    "BMI": bmi,
    "diabetes": diabetes_status
                         
}])


## Regression 

if st.button("Estiamte 10-Year CVD Risk"):
    predicted_risk= model.predict_proba(input_df)[:,1][0]
    percentile= CHD_percentile(age, predicted_risk,CHD_df)
    action, reduction= risk_ajusted_CHD(input_df, predicted_risk)


    ## Results 

    st.subheader(f"Predicted 10-Year CVD Risk: {predicted_risk * 100:.1f}%")
    st.write("Your risk score is the percent chance that you will develop a coronary heart disease in the next 10 years")

    st.write(f"This places you at approximately the **{percentile:.1f}th percentile** " "compared to people of a similar age.")
    st.write("Below will show you some adjustments that you can make to improve your risk factor, and how these adjustments will affect your CHD risk over the next 10 years")

    ## Show most impactful change that can be made 

    st.subheader("Most Impactful Change")
    st.write( f"**{action}** could reduce your estimated 10-year risk by " f"about **{reduction * 100:.1f}%**.")




    ## Code crashes if age is imput under 30. Add if stagement here or just prevent ages under 30 from being input? 
    #  

    age_comparison= CHD_df[(CHD_df["age"]>= age-2) & (CHD_df["age"] <= age+2)]

    X_comp= age_comparison[["age", "male", "currentSmoker", "BMI", "diabetes"]]
    comp_risk= model.predict_proba(X_comp)[:,1]

    ### Risk Decomposition: Show what percentage of the CHD risk is accounted for by eaech risk factor 

    ## Model coefficients 
    
    st.subheader("What are the biggest contributers to your CVD risk factor?")
    decomp= risk_decomposition(input_df)
    st.dataframe(decomp[["feature", "percent"]].round(1), use_container_width=True)

    ## Age comparison setup 

    st.subheader("How do you compare to others your age?")


    ## Figures 1: Histogram --> Used to help person visulaize how they commpare to others their age 

    fig, ax= plt.subplots(figsize=(6,4))
    ax.hist(comp_risk, bins=20, alpha=.7)
    ax.axvline(predicted_risk, linestyle="--", label= "Your Risk")
    ax.set_xlabel("10-Year CVD Risk")
    ax.set_ylabel("Number of People")

    st.pyplot(fig)

    ## Figure 2: Risk vs Age scatter --> Helps to where their risk lies compared to the entirety of the cohort of people. Easy to see if you are above or below the trend line 

    st.subheader("Risk vs Age")

    sample= CHD_df.sample(n=min(1500, len(CHD_df)), random_state=42)
    X_sample= sample[["age", "male", "currentSmoker", "BMI", "diabetes"]]
    sample_risk= model.predict_proba(X_sample)[:,1]

    fig,ax= plt.subplots(figsize=(6,4))
    ax.scatter(sample["age"], sample_risk, alpha=.3, s=15)
    ax.scatter(age, predicted_risk, color="red", s=80, label="You")


    ## Create trend line for plots 
    z=np.polyfit(sample["age"], sample_risk, 1)
    p= np.poly1d(z)
    ax.plot(sample["age"], p(sample["age"]), linewidth=2)
    ax.set_xlabel("Age")
    ax.set_ylabel("10-Year CVD Risk")
    
    st.pyplot(fig)


    ## Plot 3: Lifestyle ajustment visualizer --> shows what quitting smoking or lowering BMI can do to improve CHD outlook 

    st.subheader("Lifestyle Change Impact on CHD")

    ajusted_CHD= []
    ajusted_CHD.append(("Current", predicted_risk))


    ## Quitting Smoking 
    if smoker_status ==1:
        s=input_df.copy()
        s["currentSmoker"]=0 
        ajusted_CHD.append(("Quit Smoking", model.predict_proba(s)[:,1][0]))

    ## Reduce BMI

    s=input_df.copy()
    s["BMI"]= CHD_df["BMI"].mean()
    ajusted_CHD.append(("Lower BMI", model.predict_proba(s)[:,1][0]))

    labels, risks= zip(*ajusted_CHD)

    fig, ax= plt.subplots(figsize=(6,4))
    ax.bar(labels, risks)
    ax.set_ylabel("10-Year CVD Risk")

    st.pyplot(fig)


    ## Plot 4: How does risk change as you age? 

    
    trajectory_base= age_risk_trajectory(input_df)
    improved_df= improved_profile(input_df)
    improved_trajectory= age_risk_trajectory(improved_df)

    fig, ax= plt.subplots(figsize=(6,4))
    ax.plot(trajectory_base["age"], trajectory_base["risk"], linewidth=2, label= "Current risk trajectory")
    ax.plot(improved_trajectory["age"], improved_trajectory["risk"], linewidth=2, linestyle="--", label= "Risk trajectory post lifetyle changes")
    ax.scatter(age, predicted_risk, color="red", zorder=5, label= "Current Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("10-Year CHD Risk")
    ax.legend()

    st.pyplot(fig)





