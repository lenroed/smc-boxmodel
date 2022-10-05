# Sequential Monte Carlo Filters for Chemical Box-Models

This python module contains classes to initiate constrained box-models and sequential monte carlo filters (particle filters) used for a study currently submitted to Atmospheric Measurement and Techniques (AMT) https://www.atmospheric-measurement-techniques.net/  
Please cite the article (citation will be added once published) if you use this code for your studies.

The implementation is based on the calculations from  
>  Gordon, N. J., Salmond, D. J., and Smith, A. F.: Novel approach to nonlinear/non-Gaussian Bayesian state estimation, in: IEE Proceedings F-radar and signal processing, vol. 140, pp. 107–113, IET, https://doi.org/10.1049/ip-f-2.1993.0015, 1993.  

>  Pitt, M. K. and Shephard, N.: Filtering via simulation: Auxiliary particle filters, Journal of the American statistical association, 94, 590–599, https://doi.org/10.1080/01621459.1999.10474153, 1999.  

Refer to the main article and other references given there for more information.  

The value for $k_{O_3,NO}$ is taken from  
>  Atkinson, R., Baulch, D. L., Cox, R. A., Crowley, J. N., Hampson, R. F., Hynes, R. G., Jenkin, M. E., Rossi, M. J., and Troe, J.: Evaluated kinetic and photochemical data for atmospheric chemistry: Volume I - gas phase reactions of Ox, HOx, NOx and SOx species, Atmos. Chem. Phys., 4, 1461–1738, https://doi.org/10.5194/acp-4-1461-2004, 2004.  
