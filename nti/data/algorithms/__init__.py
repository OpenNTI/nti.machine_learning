from nti.data.algorithms.clustering.density import DBScan
from nti.data.algorithms.clustering.density import Entropic

from nti.data.algorithms.clustering.geometric import KMeans

from nti.data.algorithms.supervised.support_vector_machine import SupportVectorMachine

DB_SCAN = DBScan.__name__
KMEANS = KMeans.__name__
ENTROPY = Entropic.__name__
SVM = SupportVectorMachine.__name__