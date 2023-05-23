def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_value, target, target_value,):
  t_subset = up_table_subset(table, target, 'equals', target_value)  #now have table with only those with Flu
  e_list = up_get_column(t_subset, evidence)         #evidence column values taken from subsetted table
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)          #percentage of values in evidence column that match evidence value
  return p_b_a + .01

def cond_probs_product(table, evidence_values, target_column, target_val):
  cond_prob_list = []
  table_columns = up_list_column_names(table)
  evidence_columns = table_columns[:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence_values)
  for evidence, value in evidence_complete:
    a = cond_prob(table, evidence, value, target_column, target_val)
    cond_prob_list += [a]
  partial_numerator = up_product(cond_prob_list)  #new puddles function
  return partial_numerator

def prior_prob(table, target, target_value):
  t_list = up_get_column(table, target)      #all values in target column taken from full table
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)           #percentage of values that have target_value
  return p_a

def naive_bayes(table, evidence_row, target):
  #compute P(Flu=0|...) by collecting cond_probs in a list, take the produce of the list, finally multiply by P(Flu=0)
  result1 = cond_probs_product(table, evidence_row, target, 0) * prior_prob(table, target, 0) #target replaces target

  #do same for P(Flu=1|...)
  result2 = cond_probs_product(table, evidence_row, target, 1) * prior_prob(table, target, 1)
 
  #Use compute_probs to get 2 probabilities
  neg, pos = compute_probs(result1, result2)
  
  #return your 2 results in a list
  return [neg, pos]

def test_load ():
  return 'loaded'

def metrics(zipped_predictions_list):
  assert isinstance(zipped_predictions_list, list), f'Expecting Parameter to be a list but instead is {type(zipped_predictions_list)}'
  assert all(isinstance(item, list) for item in zipped_predictions_list), f'Expecting Parameter to be a list of lists but instead is {type(zipped_predictions_list)}'
  assert all(isinstance(item, (tuple, list)) and len(item) == 2 for item in zipped_predictions_list), 'Expecting Parameter to contain a pair of zipped items'
  assert all(isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(value, int) for value in item) for item in zipped_predictions_list), 'Expecting Parameter to contain pairs of ints, instead given non-ints'
  assert all(isinstance(item, (list, tuple)) and all(value >= 0 for value in item) for item in zipped_predictions_list), "Expecting Parameter to contain pairs of positive values, instead given negative values"
  #Unable to return assert for zipped list error
  for pair in zipped_predictions_list:
    tn = sum([1 if pair==[0,0] else 0 for pair in zipped_predictions_list])
    tp = sum([1 if pair==[1,1] else 0 for pair in zipped_predictions_list])
    fp = sum([1 if pair==[1,0] else 0 for pair in zipped_predictions_list])
    fn = sum([1 if pair==[0,1] else 0 for pair in zipped_predictions_list])
    pop = tn+tp+fp+fn
  Accuracy = (tp + tn)/pop if pop>0 else 0
  Precision = tp/(tp+fp) if tp+fp>0 else 0
  Recall = tp/(tp+fn) if tp+fn>0 else 0
  F1 = (2*Precision*Recall)/(Precision+Recall) if Precision+Recall > 0 else 0
  return {'Precision': Precision, 'Recall': Recall, 'F1': F1, 'Accuracy': Accuracy}
