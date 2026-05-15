import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score

data =  {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'exam_score':    [40, 50, 55, 65, 70, 75, 80, 85, 90, 95]
}

df = pd.DataFrame(data)
print("Step 1- Dataset: ")
print(df)
print(df.shape)

#grouping data where x is considered a table and y is a column within the table
x = df[['hours_studied']]
y = df['exam_score']

print("\nStep 2- Input x: ")
print(x)
print("\nOutput y: ")
print(y)

#splitting data 
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2 , random_state=42)
print(f"Training rows: {len(x_train)}")
print(f"Testing rows: {len(x_test)}")

#training the model
model = LinearRegression()
model.fit(x_train , y_train)
print("model trained succesfully")

#printing out what the model learned using the formula y = mx + c
print(f"Slope (m): {model.coef_[0]:.2f}")
print(f"Intercept (c): {model.intercept_:.2f}")

#making predictions
predictions = model.predict(x_test)
print(f"Predictions: {predictions}")
print(f"Actual: {y_test.values}")

#checking the margin of error
mse = mean_squared_error(y_test , predictions)
r2 = r2_score(y_test , predictions)
print(f"MSE: {mse:.2f}")
print(f"R2: {r2:.2f}")

#appending a new value for model to predict
new_student = pd.DataFrame({'hours_studied': [11]})
prediction = model.predict(new_student)
print(f"11 hour-study might score: {prediction[0]:.2f}")