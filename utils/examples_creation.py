import os
import json
import pandas as pd
import numpy as np

def fun_reframe(df_input, prev_comments):
    history = df_input
    result = {key:history[max(-prev_comments + i, 0): i + 1]
            for key, i in enumerate(range( prev_comments, len(history)))}
    return result

def fun_expandex(df_input):
    new_dict = {df_input.name: df_input['examples']}
    df_input['new_example'] = new_dict
    return df_input

def fun_test(df_input, prev_comments):
    df_nested  = pd.DataFrame(df_input['new_example'])
    if (len(df_nested)> 0):
        author = str(df_input.name)
        comment_columns = ['t-'+str( prev_comments-i) for i in range(0,prev_comments)]+ ['t']
        df_examples_new = pd.DataFrame(columns= comment_columns )
        df_examples_new[ comment_columns] = pd.DataFrame(df_nested[df_input.name].tolist(), index= df_nested.index)
        df_examples_new.insert(0,'author', author)
        df_input['track'] = df_examples_new.to_dict(orient = 'index')
    return df_input

def main():
    #Changing filepaths
    print('current directory: ', os.getcwd() )

    ## NEW LAB PATH
    output_folder = '/home/kishore/kishore_data/outlets'
    data_folder = '/home/kishore/fan_backup/Old_code/news/outlets'

    # ##SABINE PATH 
    # output_folder = '/project/mukherjee/kishore/news_code/output'
    # data_folder = '/project/mukherjee/kishore/news_code/outlets'
    outlet = 'NewYorkTimes'
    output_folder = os.path.join( output_folder, outlet)
    data_folder = os.path.join( data_folder, outlet)

    os.chdir(output_folder)
    print('New directory: ', os.getcwd())

    #read reference files
    minimum_history = -1
    prev_comments = 12
    file_authors_pkl = os.path.join(output_folder, 'pickle_inputs/frequent_author_record.pkl')
    df_authors = pd.read_pickle(file_authors_pkl )
    df_authors.index = df_authors.index.astype(int)

    # global df_authors_mod
    #reducing data
    if minimum_history > 0:
        df_authors_mod = df_authors[ df_authors['comments'].apply(len) > minimum_history]
    else:
        df_authors_mod = df_authors

    df_authors_mod['examples'] = df_authors_mod['comments'].apply(fun_reframe, args = (prev_comments,))
    df_authors_mod  = df_authors_mod.apply(fun_expandex, axis = 1)

    df_authors_mod = df_authors_mod.apply(fun_test, axis =1, args = (prev_comments,))

    df_final = pd.DataFrame()
    for each in df_authors_mod['track'].values:
        if each is not np.NaN:
            df_final = pd.concat( [df_final, 
                pd.DataFrame.from_dict( each, orient = 'index')])

    df_final.reset_index(drop = True, inplace = True)
    # #pickle dataset
    # file_examples = os.path.join(output_folder, 'shortbert_inputs/examples.pkl')
    # df_final.to_pickle(file_examples)
    

if __name__ == '__main__':
    main()