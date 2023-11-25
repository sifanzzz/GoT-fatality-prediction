import pandas as pd
import pytest
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.preprocess import preprocess_inputs

test_df = pd.DataFrame({'S.No': {0: 1, 1: 2,2: 3,3: 4,4: 5,5: 6,6: 7,7: 8,8: 9,9: 
                    10,10: 11,11: 12,12: 13,13: 14,14: 15},
 'plod': {0: 0.946,1: 0.613,2: 0.507,3: 0.924,4: 0.383,5: 0.979,6: 0.986,7: 0.964,
        8: 0.276,9: 0.609,10: 0.833,11: 0.015,12: 0.187,13: 0.253,14: 0.295},

 'name': {0: 'Viserys II Targaryen',1: 'Walder Frey',2: 'Addison Hill',3: 'Aemma Arryn',
            4: 'Sylva Santagar',5: 'Tommen Baratheon',6: 'Valarr Targaryen',7: 'Viserys I Targaryen',
            8: 'Wilbert',9: 'Wilbert Osgrey',10: 'Will',11: 'Will (orphan)',12: 'Will (squire)',
            13: 'Will (Standfast)',14: 'Will (Treb)'},

 'title': {0: None,1: 'Lord of the Crossing',2: 'Ser',3: 'Queen',4: 'Greenstone',5: None,6: 'Hand of the King',
            7: None,8: 'Ser',9: 'Ser',10: None,11: None,12: None,13: None,14: None},

 'male': {0: 1,1: 1,2: 1,3: 0,4: 0,5: 1,6: 1,7: 1,8: 1,9: 1,10: 1,11: 0,12: 1,13: 0,14: 1},
 'culture': {0: None,1: 'Rivermen',2: None,3: None,4: 'Dornish',5: None,6: 'Valyrian',7: None,
            8: None,9: None,10: None,11: None,12: None,13: None,14: None},

 'dateOfBirth': {0: None,1: 208.0,2: None,3: 82.0,4: 276.0,5: None,6: 183.0,7: None,
                8: None,9: None,10: None,11: None,12: None,13: None,14: None},

 'DateoFdeath': {0: None,1: None,2: None,3: 105.0,4: None,5: None,6: 209.0,7: None,
                8: 298.0,9: None,10: 297.0,11: None,12: None,13: None,14: None},

 'mother': {0: 'Rhaenyra Targaryen',1: None,2: None,3: None,4: None,5: 'Cersei Lannister',6: None,
            7: 'Alyssa Targaryen',8: None,9: None,10: None,11: None,12: None,13: None,14: None},

 'father': {0: 'Daemon Targaryen',1: None,2: None,3: None,4: None,5: 'Robert Baratheon',6: None,
            7: 'Baelon Targaryen',8: None,9: None,10: None,11: None,12: None,13: None,14: None},

 'heir': {0: 'Aegon IV Targaryen',1: None,2: None,3: None,4: None,5: 'Myrcella Baratheon',6: None,
            7: 'Rhaenyra Targaryen',8: None,9: None,10: None,11: None,12: None,13: None,14: None},

 'house': {0: None,1: 'House Frey',2: 'House Swyft',3: 'House Arryn',4: 'House Santagar',5: None,
            6: 'House Targaryen',7: None,8: None,9: 'House Osgrey',10: "Night's Watch",11: None,
            12: None,13: 'House Osgrey',14: 'House Osgrey'},

 'spouse': {0: None,1: 'Perra Royce',2: None,3: 'Viserys I Targaryen',4: 'Eldon Estermont',5: None,
            6: 'Kiera of Tyrosh',7: None,8: None,9: None,10: None,11: None,12: None,13: None,14: None},

 'book1': {0: 0,1: 1,2: 0,3: 0,4: 0,5: 0,6: 0,7: 0,8: 0,9: 0,10: 1,11: 0,12: 0,13: 0,14: 0},

 'book2': {0: 0,1: 1,2: 0,3: 0,4: 0,5: 0,6: 0,7: 0,8: 0,9: 0,10: 1,11: 0,12: 0,13: 0,14: 0},

 'book3': {0: 0,1: 1,2: 0,3: 0,4: 0,5: 0,6: 0,7: 0,8: 1,9: 0,10: 0,11: 0,12: 0,13: 0,14: 0},

 'book4': {0: 0,1: 1,2: 1,3: 0,4: 1,5: 0,6: 0,7: 0,8: 0,9: 0,10: 0,11: 1,12: 0,13: 0,14: 0},

 'book5': {0: 0,1: 1,2: 0,3: 0,4: 0,5: 0,6: 0,7: 0,8: 0,9: 0,10: 0,11: 0,12: 0,13: 0,14: 0},

 'isAliveMother': {0: 1.0,1: None,2: None,3: None,4: None,5: 1.0,6: None,7: 1.0,8: None,9: None,
                    10: None,11: None,12: None,13: None,14: None},

 'isAliveFather': {0: 0.0,1: None,2: None,3: None,4: None,5: 1.0,6: None,7: 1.0,8: None,9: None,
                    10: None,11: None,12: None,13: None,14: None},

 'isAliveHeir': {0: 0.0,1: None,2: None,3: None,4: None,5: 1.0,6: None,7: 1.0,8: None,9: None,
                10: None,11: None,12: None,13: None,14: None},

 'isAliveSpouse': {0: None,1: 1.0,2: None,3: 0.0,4: 1.0,5: None,6: 1.0,7: None,8: None,9: None,
                    10: None,11: None,12: None,13: None,14: None},

 'isMarried': {0: 0,1: 1,2: 0,3: 1,4: 1,5: 0,6: 1,7: 0,8: 0,9: 0,10: 0,11: 0,12: 0,13: 0,14: 0},

 'isNoble': {0: 0,1: 1,2: 1,3: 1,4: 1,5: 0,6: 1,7: 0,8: 1,9: 1,10: 0,11: 0,12: 0,13: 0,14: 0},

 'age': {0: None,1: 97.0,2: None,3: 23.0,4: 29.0,5: None,6: 26.0,7: None,8: None,9: None,
        10: None,11: None,12: None,13: None,14: None},

 'numDeadRelations': {0: 11,1: 1,2: 0,3: 0,4: 0,5: 5,6: 0,7: 5,8: 0,9: 0,10: 0,11: 0,12: 0,13: 0,14: 0},

 'boolDeadRelations': {0: 1,1: 1,2: 0,3: 0,4: 0,5: 1,6: 0,7: 1,8: 0,9: 0,10: 0,11: 0,12: 0,13: 0,14: 0},

 'isPopular': {0: 1,1: 1,2: 0,3: 0,4: 0,5: 1,6: 1,7: 1,8: 0,9: 0,10: 0,11: 0,12: 0,13: 0,14: 0},

 'popularity': {0: 0.605351171,1: 0.89632107,2: 0.267558528,3: 0.183946488,4: 0.043478261,5: 1.0,
                6: 0.431438127,7: 0.678929766,8: 0.006688963,9: 0.02006689,10: 0.163879599,11: 0.003344482,
                12: 0.003344482,13: 0.003344482,14: 0.003344482},

 'isAlive': {0: 0,1: 1,2: 1,3: 0,4: 1,5: 1,6: 0,7: 0,8: 0,9: 1,10: 0,11: 1,12: 1,13: 1,14: 1}})

def test_drop_cols():
    expected_drop_cols = ['S.No', 'plod', 'name', 'heir', 'isAliveMother', 'isAliveFather',
                          'isAliveHeir', 'isAliveSpouse', 'father', 'mother', 'spouse',
                          'age', 'DateoFdeath']
    processed_df = preprocess_inputs(test_df, {}, expected_drop_cols, [])
    assert set(expected_drop_cols).isdisjoint(processed_df.columns)

def test_fill_nan():
    expected_fill_nan_cols = {
        'title': 'No Title',
        'house': 'No House',
        'culture': 'Unknown'
    }
    processed_df = preprocess_inputs(test_df, expected_fill_nan_cols, [], [])
    for key, val in expected_fill_nan_cols.items():
        assert processed_df[key].fillna(val).equals(processed_df[key])

def test_columns_existence_after_preprocessing():
    fill_nan_cols = {'title':'No Title', 'house':'No House', 'dateOfBirth':'dateOfBirth', 'culture':'Unknown'}
    drop_cols = ['S.No', 'plod', 'name', 'heir', 'isAliveMother', 'isAliveFather', 'isAliveHeir', 'isAliveSpouse',
                 'father', 'mother', 'spouse', 'age', 'DateoFdeath']
    less_frequent_cols = ['title', 'culture', 'house']

    processed_df = preprocess_inputs(test_df, fill_nan_cols, drop_cols, less_frequent_cols)

    expected_cols = list(set(test_df.columns) - set(drop_cols))
    for col in expected_cols:
        assert col in processed_df.columns

if __name__ == "__main__":
    pytest.main()
