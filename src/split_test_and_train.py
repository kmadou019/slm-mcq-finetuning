#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
def main():
    df = pd.read_csv("../data/lisa_sheets.csv")
    df_train, df_test = train_test_split(df,test_size=0.3)
    
    pd.DataFrame.to_csv(df_train,"../data/lisa_train.csv",index=False)
    pd.DataFrame.to_csv(df_test,"../data/lisa_test.csv",index=False)


if __name__ == "__main__":
    main()