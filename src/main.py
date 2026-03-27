from update_data import setup_for_predict, setup_for_eval
from user import get_user_input
from predict import make_prediction

pipe, master_df = setup_for_predict() # args(update_data, update_model)
#pipe, master_df, test_df = setup_for_eval()


while True:
    print("\nenter q to quit")
    home_input = input("enter the home team: ")
    away_input = input("enter the away team: ")

    if home_input == "q" or away_input == "q":
        break
    
    home_alias, away_alias = get_user_input(home_input, away_input)
    make_prediction(home_alias, away_alias, pipe, master_df)
    print("---------------------------------------------------")
