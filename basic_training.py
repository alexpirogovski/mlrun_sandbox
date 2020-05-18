from mlrun import get_or_create_ctx
from mlrun.artifacts import ChartArtifact, TableArtifact
import pandas as pd
from random import randint
from sklearn.linear_model import LinearRegression


def my_job(context, test_data, train_set_count):
    # load MLRUN runtime context (will be set by the runtime framework e.g. KubeFlow)

    # get parameters from the runtime context (or use defaults)

    # access input metadata, values, files, and secrets (passwords)
    print(f'Run: {context.name} (uid={context.uid})')
    print(f'Params: p1={test_data}, p2={train_set_count}')
#     print('accesskey = {}'.format(context.get_secret('ACCESS_KEY')))
#     print('file\n{}\n'.format(context.get_input('infile.txt', 'infile.txt').get()))
    
    # Here comes AlexP's genious code
    

    TRAIN_SET_LIMIT = 1000
    TRAIN_SET_COUNT = train_set_count

    TRAIN_INPUT = list()
    TRAIN_OUTPUT = list()
    for i in range(TRAIN_SET_COUNT):
        a = randint(0, TRAIN_SET_LIMIT)
        b = randint(0, TRAIN_SET_LIMIT)
        c = randint(0, TRAIN_SET_LIMIT)
        op = a + (2*b) + (3*c)
        TRAIN_INPUT.append([a, b, c])
        TRAIN_OUTPUT.append(op)
        
    
    predictor = LinearRegression(n_jobs=-1)
    predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)
    
    X_TEST = [test_data]
    outcome = predictor.predict(X=X_TEST)
    coefficients = predictor.coef_

#     print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))
    context.log_result('Outcome', outcome)
    deviation = []
    for c in coefficients:
        deviation.append(abs(c - int(c)))
    context.log_result('deviation', max(deviation))
    context.log_result('Coefficients', coefficients)
            
    
        
    
#     # Run some useful code e.g. ML training, data prep, etc.

#     # log scalar result values (job result metrics)
#     context.log_result('accuracy', p1 * 2)
#     context.log_result('loss', p2 * 3)
#     context.set_label('framework', 'sklearn')

#     # log various types of artifacts (file, web page, table), will be versioned and visible in the UI
#     context.log_artifact('model', body=b'Omlette is a silly cat', local_path='model.txt', labels={'framework': 'xgboost'})
#     context.log_artifact('html_result', body=b'<b> Some HTML <b>', local_path='result.html')
#     context.log_artifact(TableArtifact('dataset', '1,2,3\n4,5,6\n', visible=True,
#                                         header=['A', 'B', 'C']), local_path='dataset.csv')

#     # create a chart output (will show in the pipelines UI)
#     chart = ChartArtifact('chart')
#     chart.labels = {'type': 'roc'}
#     chart.header = ['Epoch', 'Accuracy', 'Loss']
#     for i in range(1, 8):
#         chart.add_row([i, i/20+0.75, 0.30-i/20])
#     context.log_artifact(chart)

#     raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
#                 'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
#                 'age': [42, 52, 36, 24, 73],
#                 'testScore': [25, 94, 57, 62, 70]}
#     df = pd.DataFrame(raw_data, columns=[
#         'first_name', 'last_name', 'age', 'testScore'])
#     context.log_dataset('mydf', df=df, stats=True)


if __name__ == "__main__":
    context = get_or_create_ctx('alexp_train')
    p1 = context.get_param('test_data')
    p2 = context.get_param('train_set_count')
    my_job(context, p1, p2)
    
