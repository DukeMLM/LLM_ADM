import pandas as pd
import numpy as np
import json
def trans_to_llm_data(mode,data_size):

    ## create prompt
    file_path = f'/home/dl370/data/LLM/150_350THz_dataset/g_{mode}_150_350THz_f50_d{data_size}.csv'
    data_x = pd.read_csv(file_path,header=None)
    
    formatted_questions7 = []

    for index, row in data_x.iterrows():
        question = (
            f" The All-dielectric metasurface suspend in free space is: <{round(float(row.iloc[0]), 3)}, {round(float(row.iloc[1]), 3)}, {round(float(row.iloc[2]),3)}, {round(float(row.iloc[3]),3)},"
            f" {round(float(row.iloc[4]), 3)}, {round(float(row.iloc[5]), 3)}, {round(float(row.iloc[6]),3)}, {round(float(row.iloc[7]),3)},"
            f" {round(float(row.iloc[8]), 3)}, {round(float(row.iloc[9]), 3)}, {round(float(row.iloc[10]),3)}, {round(float(row.iloc[11]),3)},"
            f" {round(float(row.iloc[12]), 3)}, {round(float(row.iloc[13]), 3)}> Get the absorptivity ###"
        )
        formatted_questions7.append(question)

    ## create completion
    file_pathy = f'/home/dl370/data/LLM/150_350THz_dataset/s_{mode}_150_350THz_f50_d{data_size}.csv'
    data_y = pd.read_csv(file_pathy,header=None)

    formatted_answers = []
    for index, row in data_y.iterrows():
        answer = np.array(row)
        # Round each element in the array to 3 decimal places
        rounded_answer = np.around(answer, decimals=3)
        formatted_answers.append(rounded_answer ) 
    print(len((formatted_answers[1])))


    option = [7]
    questions = [ formatted_questions7]
    for index2 , (option, question) in enumerate(zip(option, questions)):
        pairs = zip(question, formatted_answers)

        output_file_path = f'data_{mode}_{data_size}_1535_output50_option{option}_1.jsonl'
        print(output_file_path)

        system_input = ''

        # Writing to the JSONL file
        with open(output_file_path, 'w') as file:
            
            for index, (question, answer) in enumerate(pairs):
                # Ensure both question and answer are strings
                question_str = str(question)
                answer_str=' ['
                number=' '.join(map(lambda x: f"{x}", answer))
                answer_str += number
                # answer_str +=']'
                answer_str +=']@@@'

                # Create a JSON object for each pair
                formatted_question = {
                    "messages": [
                        {
                            "role": "system",
                            # "content": "You now are a CST simulation solver."
                            "content": system_input
                        },
                        {
                            "role": "user",
                            "content": question_str
                        },
                        {
                            "role": "assistant",
                            "content": answer_str
                        }
                    ]
                }
                # Convert the dictionary to a JSON string
                json_object = json.dumps(formatted_question)
                file.write(json_object + '\n')

if __name__ == "__main__":
    mode = 'train'
    data_size = '1000'
    trans_to_llm_data(mode,data_size)
    print('finished')