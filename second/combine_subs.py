import fire
import pandas as pd


def two(path1, path2):
    sub1 = pd.read_csv(path1)
    sub2 = pd.read_csv(path1)

    print(sub1.isna().sum())
    for idx in sub1.loc[sub1['PredictionString'].isna()].index.tolist():
        token = sub1.loc[idx]['Id']
        sub1.loc[idx]['PredictionString'] = sub2.loc[sub2['Id']==token]
    print(sub1.isna().sum())
    sub1.to_csv('combined.csv', index=False)

if __name__ == "__main__":
    fire.Fire()
