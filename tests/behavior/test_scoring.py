from ai_cdss.data_processor import DataProcessor
from ai_cdss.data_loader import DataLoaderMock
from IPython.display import display

def test_prescriptions_inference():

    loader = DataLoaderMock(
        num_patients=1,
        num_protocols=5,
        num_sessions=10
    )

    sess = loader.load_session_data()
    print("\n")
    display(sess.iloc[:,15:24])

    ts = loader.load_timeseries_data()
    ppf = loader.load_ppf_data()
    init = loader.load_protocol_init()

    processor = DataProcessor(
        weights=[1,1,1],
        alpha=0.5
    )

    score = processor.process_data(sess, ts, ppf, init)
    print(score)

    ##### CHECK PRESCRIPTIONS BEHAVIOR
    ##### - When prescriptions is canceled PRESCRIPTION_ENDING_DATE changes in all rows?

    # display(score)