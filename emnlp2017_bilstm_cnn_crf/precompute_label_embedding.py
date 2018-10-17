from util.preprocessing import  build_vocab, build_indexes_from_domains, build_matrix_from_domains, get_nearest_labels
from util.constants import  TASKS, NERS
import pickle

word_count = build_vocab(TASKS + NERS)
idxs = build_indexes_from_domains(TASKS + NERS, word_count)
matrix = build_matrix_from_domains(TASKS + NERS, idxs, word_count)

get_nearest_labels('ATIS', ['MIT_Restaurant','MIT_Movie', 'OntoNotes_NW','CONLL_2003_NER'], matrix, idxs)

pickle.dump(matrix, open("label_embedding.emb", "wb"))
pickle.dump(idxs, open("label_embedding.idxs", "wb"))

matrix =  pickle.load(open("label_embedding.emb", "rb"))
idxs = pickle.load(open("label_embedding.idxs", "rb"))
get_nearest_labels('ATIS', ['MIT_Restaurant','MIT_Movie', 'OntoNotes_NW','CONLL_2003_NER'], matrix, idxs)


