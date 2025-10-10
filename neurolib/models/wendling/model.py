from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class WendlingModel(Model):
    """
    Wendling-Chauvel Neural Mass Model (10-ODE system).
    
    This model represents a cortical column with pyramidal cells, excitatory interneurons,
    slow inhibitory interneurons (GABA_A, somatic), and fast inhibitory interneurons 
    (GABA_A, dendritic).
    
    References:
    - Wendling, F., Bartolomei, F., Bellanger, J. J., & Chauvel, P. (2002). 
      Epileptic fast activity can be explained by a model of impaired GABAergic 
      dendritic inhibition. European Journal of Neuroscience, 15(9), 1499-1508.
    - Köksal Ersöz, E., et al. (2020). Neural mass modeling of slow-fast dynamics 
      of seizure initiation and abortion. PLoS Computational Biology, 16(11), e1008430.
    """

    name = "wendling"
    description = "Wendling-Chauvel Neural Mass Model"

    init_vars = [
        "y0_init", "y1_init", "y2_init", "y3_init", "y4_init",
        "y5_init", "y6_init", "y7_init", "y8_init", "y9_init"
    ]
    
    state_vars = [
        "y0", "y1", "y2", "y3", "y4",  # Membrane potentials
        "y5", "y6", "y7", "y8", "y9"   # Derivatives
    ]
    
    output_vars = ["y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9"]
    default_output = "y1"  # Primary output, can compute v_pyr = y1 - y2 - y3

    def __init__(self, params=None, Cmat=None, Dmat=None, seed=None, sigmoid_type="wendling2002"):
        """
        Initialize Wendling model.
        
        :param params: Parameter dictionary, defaults to None
        :type params: dict, optional
        :param Cmat: Structural connectivity matrix, defaults to None
        :type Cmat: numpy.ndarray, optional
        :param Dmat: Fiber length matrix, defaults to None
        :type Dmat: numpy.ndarray, optional
        :param seed: Random seed, defaults to None
        :type seed: int, optional
        :param sigmoid_type: Sigmoid variant ("wendling2002" or "pcbi2020"), defaults to "wendling2002"
        :type sigmoid_type: str, optional
        """
        
        self.Cmat = Cmat
        self.Dmat = Dmat
        self.seed = seed
        self.sigmoid_type = sigmoid_type
        
        # Integration function
        integration = ti.timeIntegration
        
        # Load default parameters
        if params is None:
            params = dp.loadDefaultParams(
                Cmat=self.Cmat, 
                Dmat=self.Dmat, 
                seed=self.seed,
                sigmoid_type=self.sigmoid_type
            )
        
        # Initialize base class
        super().__init__(integration=integration, params=params)
    
    def get_output_signal(self):
        """
        Compute the pyramidal output signal: v_pyr = y1 - y2 - y3.
        
        This is the typical EEG/LFP surrogate signal.
        
        :return: Output signal for each node
        :rtype: numpy.ndarray
        """
        if hasattr(self, 'y1') and hasattr(self, 'y2') and hasattr(self, 'y3'):
            return self.y1 - self.y2 - self.y3
        else:
            raise ValueError("Model has not been run yet. Call model.run() first.")
