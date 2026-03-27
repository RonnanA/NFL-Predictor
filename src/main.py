from update_data import setup_for_predict, setup_for_eval
from user import get_user_input
from predict import make_prediction

pipe, snapshots_df = setup_for_predict(update_data=True)
# start from scratch
#pipe, snapshots_df = setup_for_predict(update_model=True)
#model evaluation mode
#pipe, snapshots_df, cv_results = setup_for_eval()

while True:
    print("\nenter q to quit")
    home_input = input("enter the home team: ")
    away_input = input("enter the away team: ")

    if home_input == "q" or away_input == "q":
        break

    home_alias, away_alias = get_user_input(home_input, away_input)
    result = make_prediction(home_alias, away_alias, pipe, snapshots_df)

    if "error" in result:
        print(f"error: {result['error']}")
    else:
        print(f"\n  {result['away_team']} @ {result['home_team']}")
        print(f"  predicted winner:  {result['predicted_winner']}")
        print(f"  home win prob:     {result['home_win_prob']:.1%}")
        print(f"  confidence:        {result['confidence']:.1%}")

    print("---------------------------------------------------")