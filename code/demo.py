import os
os.system('pip install composeml')

# Will a customer spend more than 300 in the next hour of transactions?
import composeml as cp
df = cp.demos.load_transactions()
df = df[df.columns[:7]]
print(df.sample(10))

# First, we represent the prediction problem with a labeling function and a label maker.
def total_spent(ds):
    return ds['amount'].sum()

label_maker = cp.LabelMaker(
    target_entity="customer_id",
    time_index="transaction_time",
    labeling_function=total_spent,
    window_size="1h",
)

# Then, we run a search to automatically generate the training examples.
label_times = label_maker.search(
    df.sort_values('transaction_time'),
    num_examples_per_instance=2,
    minimum_data='2014-01-01',
    drop_empty=False,
    verbose=False,
)

print('label_times in raw')
print(label_times.sample(10))

label_times = label_times.threshold(300)
print('label_times meet 300$')
print(label_times.sample(10))