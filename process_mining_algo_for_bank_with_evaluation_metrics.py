"""
Χρήση παραδοσιακών αλγορίθμων process mining.
"""
import pm4py
from pm4py.algo.filtering.log.timestamp import timestamp_filter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petrinet import visualizer, parameters
from pm4py.algo.conformance.alignments import algorithm as alignment
from pm4py.objects.petri.align_utils import pretty_print_alignments
from contextlib import redirect_stdout
from pm4py.evaluation.simplicity import evaluator as simplicity_factory
from pm4py.evaluation.precision import evaluator as precision_evaluator
from pm4py.evaluation.replay_fitness import evaluator as fitness_evaluator
log = pm4py.read_xes(r'C:\Users\user\Desktop\ΔΙΠΛΩΜΑΤΙΚΗ\12696884\BPI Challenge 2017.xes\BPI Challenge 2017.xes')

log_2 = log[:1000]
#log = log[:1000] Για να βρώ το Precision παιρνω 1000 traces γιατί αλλιώς κάνει πάνω από 2 ώρες για να το βρεί.
"""
ALPHA-MINER
"""
alpha_petri, initial_marking, final_marking = alpha_miner.apply(log)
gviz= visualizer.apply(alpha_petri, initial_marking, final_marking)
visualizer.view(gviz)

alpha_petri_2, initial_marking, final_marking = alpha_miner.apply(log_2) # Για το precision
"""
HEURISTIC-MINER
"""
heuristic_petri, initial_marking, final_marking = heuristics_miner.apply(log)
#show petri net
print('Heuristics Miner PetriNet\n')
gviz= visualizer.apply(heuristic_petri, initial_marking, final_marking)
visualizer.view(gviz)

heuristic_petri_2, initial_marking, final_marking = heuristics_miner.apply(log_2) # Για το precision
"""
HEURISTIC-MINER WITH PARAMETERS
"""
params = {"dependency_thresh": 0.85}
heuristic_with_params_petri, initial_marking, final_marking = heuristics_miner.apply(log, parameters=params)
#show petri net
print('Heuristics Miner PetriNet\n')
gviz= visualizer.apply(heuristic_with_params_petri, initial_marking, final_marking)
visualizer.view(gviz)

"""
INDUCTIVE-MINER
"""
inductive_petri, initial_marking, final_marking = inductive_miner.apply(log)
gviz= visualizer.apply(inductive_petri, initial_marking, final_marking)
visualizer.view(gviz)

inductive_petri_2, initial_marking, final_marking = inductive_miner.apply(log_2) # Για το precision



"""
ALIGNMENTS
"""
net, initial_marking, final_marking = inductive_miner.apply(log[:1000])
gviz= visualizer.apply(net, initial_marking, final_marking)
alignments = alignment.apply_log(log[:1000], net, initial_marking, final_marking)

with open('C:/Users/user/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/ΔΙΠΛΩΜΑΤΙΚΗ_2/pretty_allignment.txt', 'w') as f:
    with redirect_stdout(f):
        pretty_print_alignments(alignments)
f.close()




"""
SIMPLICITY για Alpha-Miner, Inductive-Miner, Heuristic-Miner και Heuristic-Miner-With-Parameters.

RESULTS:
    simplicity_alpha= 1.0
    simplicity_inductive= 0.6274509803921569
    simplicity_heuristic= 0.5186721991701245
    simplicity_heuristic_with_params= 0.5146443514644352
"""
simplicity_alpha = simplicity_factory.apply(alpha_petri)
simplicity_inductive = simplicity_factory.apply(inductive_petri)
simplicity_heuristic = simplicity_factory.apply(heuristic_petri)
simplicity_heuristic_with_params = simplicity_factory.apply(heuristic_with_params_petri)

print("simplicity_alpha=",simplicity_alpha)
print("simplicity_inductive=",simplicity_inductive)
print("simplicity_heuristic=",simplicity_heuristic)
print("simplicity_heuristic_with_params=",simplicity_heuristic_with_params)




"""
FITNESS για Alpha-Miner και Inductive-Miner.
"""


fitness_alpha = fitness_evaluator.apply(log, alpha_petri, initial_marking, final_marking)
print("fitness_alpha=",fitness_alpha)

# Για inductive και heuristic ο υπολογισμός χρειάζεται πάρα πολύ ώρα.
#fitness_heuristic = fitness_evaluator.apply(log, heuristic_petri, initial_marking, final_marking)
#print("fitness_heuristic=",fitness_heuristic)


"""
PRECISION
"""

alpha_prec = precision_evaluator.apply(log_2, alpha_petri, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
alpha_prec #0.08760792937295869

heuristic_prec = precision_evaluator.apply(log_2, heuristic_petri, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
heuristic_prec #0.749585222873576

inductive_prec = precision_evaluator.apply(log_2, inductive_petri, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
inductive_prec

