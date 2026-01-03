Dose–Response Model

The dose–response assessment describes the quantitative relationship between the ingested dose of *Escherichia coli* and the probability of infection in exposed individuals. In this study, a β-Poisson dose–response model was employed, which is widely used in microbial risk assessment for enteric pathogens and is recommended in established QMRA guidelines.

The β-Poisson model expresses the probability of infection, (P_{inf}), as a function of the ingested dose, (N), according to:

[
P{inf,day} = 1 - (1 + ((N/N{50})*((2^(1/alpha)) - 1)))^(-alpha)
]

where (alpha) is a dimensionless infectivity parameter and (N_{50}) represents the median infectious dose corresponding to a 50% probability of infection. Together, these parameters characterize pathogen virulence and host–pathogen interaction under the assumed exposure conditions.

The β-Poisson framework was selected due to its flexibility in representing a wide range of dose–response behaviors and its extensive application in QMRA studies involving *E. coli* and other enteric bacteria. Model parameters were adopted from peer-reviewed literature to ensure consistency with established dose–response relationships and comparability with previous risk assessments.

At this stage of the methodology, the dose–response model defines the mathematical relationship between dose and infection probability only. Consideration of variability, uncertainty, and parameter sampling is addressed separately within the probabilistic risk modeling framework described in subsequent sections.
