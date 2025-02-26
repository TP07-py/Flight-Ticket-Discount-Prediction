# Flight-Ticket-Discount-Prediction

This project predicts whether a flight passenger is eligible for a discount based on various factors such as age, gender, departure month, ticket price, and country.

## Features
- Merges airport country mapping with passenger data
- Extracts departure day, month, and year from the departure date
- Applies discount rules based on:
  - Age (young kids and seniors)
  - Specific months (December & January)
  - Ticket price thresholds by gender
  - Passengers from select countries (USA, UK, Canada)
- Uses a Decision Tree Classifier for predictive modeling
- Saves predictions to a CSV file

## Requirements
Ensure you have Python installed along with the following dependencies:
- `pandas`
- `numpy`
- `sklearn`

## How It Works
1. Reads the datasets:
   - `prediction_challenge_train.csv` (training data)
   - `airport_country_code_mapping.csv` (airport country mapping)
   - `prediction_challenge_test.csv` (test data for prediction)
2. Cleans and preprocesses the data by extracting relevant features.
3. Implements predefined discount rules based on demographics and travel details.
4. Trains a Decision Tree Classifier on the training dataset.
5. Uses the trained model to predict missing discount statuses.
6. Saves the final predictions to `predictions.csv`.

## File Structure
```
.
├── airport_country_code_mapping.csv  # Airport to country mapping
├── prediction_challenge_train.csv    # Training dataset
├── prediction_challenge_test.csv     # Test dataset for predictions
├── discount_prediction.py            # Script for discount prediction
├── predictions.csv                    # Output file with predicted discounts
```

## Future Improvements
- Enhance discount criteria using additional demographic factors
- Implement other machine learning models for better accuracy
- Create an interactive dashboard for visualization

## Contributing
If you'd like to contribute, feel free to fork the repository and submit a pull request.
