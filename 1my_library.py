def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_value, target, target_value,):
  t_subset = up_table_subset(table, target, 'equals', target_value)  #now have table with only those with Flu
  e_list = up_get_column(t_subset, evidence)         #evidence column values taken from subsetted table
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)          #percentage of values in evidence column that match evidence value
  return p_b_a

def cond_probs_product(table, evidence_values, target_column, target_val):
  cond_prob_list = []
  for evidence, value in evidence_complete:
    a = cond_prob(table, evidence, value, target_column, target_val)
    cond_prob_list += [a]
  partial_numerator = up_product(cond_prob_list)  #new puddles function
  return partial_numerator

def prior_prob(table, target, target_value):
  t_list = up_get_column(table, target)      #all values in target column taken from full table
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)           #percentage of values that have target_value
  return p_a

def test_load ():
  return 'loaded'