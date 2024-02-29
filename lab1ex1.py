
import math
from tools.venn import A, B, omega, plot_venn

# First law
plot_venn(omega - A.union(B))
plot_venn((omega-A)&(omega-B))
# Second law
plot_venn(omega- A.intersection(B))
plot_venn((omega - A)|(omega - B))