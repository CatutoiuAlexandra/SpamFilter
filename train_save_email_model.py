import model 

dataset_path = 'emails.csv'

print("Training model...")
trained_model = model.train_model(dataset_path)

model_filename = 'spam_classifier.joblib'

print("Saving model to", model_filename)
model.save_model(trained_model, model_filename)

print("Model training and saving completed.")