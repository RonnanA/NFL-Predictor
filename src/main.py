from model import test_train_sklearn, test_train_split
from merge_files import get_master_df
from features import build_features


df = get_master_df()
X_train, y_train, X_test, y_test, test_df = test_train_split(df)
model, scored_df = test_train_sklearn(X_train, y_train, X_test, y_test, test_df)

scored_df.to_csv("test.csv", index=False)