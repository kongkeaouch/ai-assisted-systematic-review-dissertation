import pandas as pd
import timeit 
from data import ASReviewData
from review import ReviewSimulate
from state import HDF5State
from analysis import Analysis
from models.classifiers import NaiveBayesClassifier
from models.query import MaxQuery
from models.balance import DoubleBalance
from models.feature_extraction import Tfidf

dir_data = 'datasets'
dir_stats = 'stats_nb_tfidf'

# Global settings
seed = 192874123
n_instances = 10
priors = [[0,0],[1,1]]

# Init model
train_model = NaiveBayesClassifier()
feature_model = Tfidf()
query_model = MaxQuery()
balance_model = DoubleBalance(random_state=seed)

# Load dataset list
df = pd.read_csv(f'{dir_data}/index.csv',dtype=str,low_memory=False)
df_std = pd.read_csv(f'{dir_stats}/std.csv',dtype=str,low_memory=False)
df_mean = pd.read_csv(f'{dir_stats}/mean.csv',dtype=str,low_memory=False)
df_max = pd.read_csv(f'{dir_stats}/max.csv',dtype=str,low_memory=False)
df_min = pd.read_csv(f'{dir_stats}/min.csv',dtype=str,low_memory=False)

for p in priors:
  dir_state = f'state_nb_tfidf_prior_{p[0]}_{p[1]}'

  for i in df.index:
    # Load data
    dataset = df.at[i,'dataset']
    data = ASReviewData.from_file(f'datasets/{dataset}.csv')
    # os.makedirs(dir_state)
    state_file = f'{dir_state}/{dataset}.h5'

    # Start the review process
    start = timeit.default_timer()
    reviewer = ReviewSimulate(
        data,
        model=train_model,
        query_model=query_model,
        balance_model=balance_model,
        feature_model=feature_model,
        n_instances=n_instances,
        n_prior_excluded=p[1],
        n_prior_included=p[0],
        init_seed=seed,
        state_file=state_file
    )
    reviewer.review()
    end = timeit.default_timer()

    # Analyse results
    state = HDF5State(state_file)
    analysis = Analysis([state])
    n_queries = state.n_queries()
    wss_95 = analysis.wss(95)[0]
    wss_100 = analysis.wss(100)[0]
    rrf_5 = analysis.rrf(5)[0]
    rrf_10 = analysis.rrf(10)[0]
    td = analysis.avg_time_to_discovery()
    atd = sum(td.values()) / int(df.at[i,'n_papers'])
    t_simulate = end - start
    settings = state.settings
    td_items = td.items()
    td_str = {int(key): int(value) for key, value in td_items}

    # Save results
    df.at[i,'n_queries'] = n_queries
    df.at[i,'wss@95'] = wss_95
    df.at[i,'wss@100'] = wss_100
    df.at[i,'rrf@5'] = rrf_5
    df.at[i,'rrf@10'] = rrf_10
    df.at[i,'atd'] = atd
    df.at[i,'td'] = td
    df.at[i,'t_simulate'] = t_simulate
    df.at[i,'settings'] = settings

    df.to_csv(f'{dir_stats}/{dir_state}.csv',index=False)

  n_papers_mean = df['n_papers'].mean()
  n_included_mean = df['n_included'].mean()
  n_queries_mean = df['n_queries'].mean()
  wss_95_mean = df['wss@95'].mean()
  wss_100_mean = df['wss@100'].mean()
  rrf_5_mean = df['rrf@5'].mean()
  rrf_10_mean = df['rrf@10'].mean()
  atd_mean = df['atd'].mean()
  t_simulate_mean = df['t_simulate'].mean()

  df_mean = df.append({
    'simulation': dir_state, 
    'n_papers': n_papers_mean, 
    'n_included': n_included_mean,
    'n_queries': n_queries_mean,
    'wss@95': wss_95_mean,
    'wss@100': wss_100_mean,
    'rrf@5': rrf_5_mean,
    'rrf@10': rrf_10_mean,
    'atd': atd_mean,
    't_simulate': t_simulate_mean
    },  
  ignore_index=True)

  df_mean.to_csv('mean.csv',index=False)

  n_papers_std = df['n_papers'].std()
  n_included_std = df['n_included'].std()
  n_queries_std = df['n_queries'].std()
  wss_95_std = df['wss@95'].std()
  wss_100_std = df['wss@100'].std()
  rrf_5_std = df['rrf@5'].std()
  rrf_10_std = df['rrf@10'].std()
  atd_std = df['atd'].std()
  t_simulate_std = df['t_simulate'].std()

  df_std = df_std.append({
    'simulation': dir_state, 
    'n_papers': n_papers_std, 
    'n_included': n_included_std,
    'n_queries': n_queries_std,
    'wss@95': wss_95_std,
    'wss@100': wss_100_std,
    'rrf@5': rrf_5_std,
    'rrf@10': rrf_10_std,
    'atd': atd_std,
    't_simulate': t_simulate_std
    },  
  ignore_index=True)

  df_std.to_csv('std.csv',index=False)

  n_papers_idxmax = df['n_papers'].idxmax()
  n_included_idxmax = df['n_included'].idxmax()
  n_queries_idxmax = df['n_queries'].idxmax()
  wss_95_idxmax = df['wss@95'].idxmax()
  wss_100_idxmax = df['wss@100'].idxmax()
  rrf_5_idxmax = df['rrf@5'].idxmax()
  rrf_10_idxmax = df['rrf@10'].idxmax()
  atd_idxmax = df['atd'].idxmax()
  t_simulate_idxmax = df['t_simulate'].idxmax()

  n_papers_max = df.at[n_papers_idxmax,'n_papers']
  n_included_max = df.at[n_included_idxmax,'n_included']
  n_queries_max = df.at[n_queries_idxmax,'n_queries']
  wss_95_max = df.at[wss_95_idxmax,'wss@95']
  wss_100_max = df.at[wss_100_idxmax,'wss@100']
  rrf_5_max = df.at[rrf_5_idxmax,'rrf@5']
  rrf_10_max = df.at[rrf_10_idxmax,'rrf@10']
  atd_max = df.at[atd_idxmax,'atd']
  t_simulate_max = df.at[t_simulate_idxmax,'t_simulate']
  n_papers_max_file = df.at[n_papers_idxmax,'file']
  n_included_max_file = df.at[n_included_idxmax,'file']
  n_queries_max_file = df.at[n_queries_idxmax,'file']
  wss_95_max_file = df.at[wss_95_idxmax,'file']
  wss_100_max_file = df.at[wss_100_idxmax,'file']
  rrf_5_max_file = df.at[rrf_5_idxmax,'file']
  rrf_10_max_file = df.at[rrf_10_idxmax,'file']
  atd_max_file = df.at[atd_idxmax,'file']
  t_simulate_max_file = df.at[t_simulate_idxmax,'file']

  df_max = df_max.append({
    'simulation': dir_state, 
    'n_papers': n_papers_max, 
    'n_included': n_included_max,
    'n_queries': n_queries_max,
    'wss@95': wss_95_max,
    'wss@100': wss_100_max,
    'rrf@5': rrf_5_max,
    'rrf@10': rrf_10_max,
    'atd': atd_max,
    't_simulate': t_simulate_max,
    'n_papers_file': n_papers_max_file,
    'n_included_file': n_included_max_file,
    'n_queries_file': n_queries_max_file,
    'wss@95_file': wss_95_max_file,
    'wss@100_file': wss_100_max_file,
    'rrf@5_file': rrf_5_max_file,
    'rrf@10_file': rrf_10_max_file,
    'atd_file': atd_max_file,
    't_simulate_file': t_simulate_max_file
    }, 
  ignore_index=True)

  df_max.to_csv('max.csv',index=False)

  n_papers_idxmin = df['n_papers'].idxmin()
  n_included_idxmin = df['n_included'].idxmin()
  n_queries_idxmin = df['n_queries'].idxmin()
  wss_95_idxmin = df['wss@95'].idxmin()
  wss_100_idxmin = df['wss@100'].idxmin()
  rrf_5_idxmin = df['rrf@5'].idxmin()
  rrf_10_idxmin = df['rrf@10'].idxmin()
  atd_idxmin = df['atd'].idxmin()
  t_simulate_idxmin = df['t_simulate'].idxmin()

  n_papers_min = df.at[n_papers_idxmin,'n_papers']
  n_included_min = df.at[n_included_idxmin,'n_included']
  n_queries_min = df.at[n_queries_idxmin,'n_queries']
  wss_95_min = df.at[wss_95_idxmin,'wss@95']
  wss_100_min = df.at[wss_100_idxmin,'wss@100']
  rrf_5_min = df.at[rrf_5_idxmin,'rrf@5']
  rrf_10_min = df.at[rrf_10_idxmin,'rrf@10']
  atd_min = df.at[atd_idxmin,'atd']
  t_simulate_min = df.at[t_simulate_idxmin,'t_simulate']
  n_papers_min_file = df.at[n_papers_idxmin,'file']
  n_included_min_file = df.at[n_included_idxmin,'file']
  n_queries_min_file = df.at[n_queries_idxmin,'file']
  wss_95_min_file = df.at[wss_95_idxmin,'file']
  wss_100_min_file = df.at[wss_100_idxmin,'file']
  rrf_5_min_file = df.at[rrf_5_idxmin,'file']
  rrf_10_min_file = df.at[rrf_10_idxmin,'file']
  atd_min_file = df.at[atd_idxmin,'file']
  t_simulate_min_file = df.at[t_simulate_idxmin,'file']

  df_min = df_min.append({
    'simulation': dir_state, 
    'n_papers': n_papers_min, 
    'n_included': n_included_min,
    'n_queries': n_queries_min,
    'wss@95': wss_95_min,
    'wss@100': wss_100_min,
    'rrf@5': rrf_5_min,
    'rrf@10': rrf_10_min,
    'atd': atd_min,
    't_simulate': t_simulate_min,
    'n_papers_file': n_papers_min_file,
    'n_included_file': n_included_min_file,
    'n_queries_file': n_queries_min_file,
    'wss@95_file': wss_95_min_file,
    'wss@100_file': wss_100_min_file,
    'rrf@5_file': rrf_5_min_file,
    'rrf@10_file': rrf_10_min_file,
    'atd_file': atd_min_file,
    't_simulate_file': t_simulate_min_file
    }, 
  ignore_index=True)

  df_min.to_csv('min.csv',index=False)