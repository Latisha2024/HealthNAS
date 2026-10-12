# # Healthcare knowledge used by the RAG system.
# # The knowledge is focused on machine-learning architecture design.

# HEALTHCARE_KNOWLEDGE = [

#     # Document 1: Chest X-ray classification
#     """
#     Chest X-ray pneumonia classification can be formulated as a
#     binary image classification problem.

#     The input consists of chest radiographic images and the target
#     classes can be normal and pneumonia.

#     Convolutional Neural Networks are suitable for this type of
#     medical image classification because convolution operations
#     can learn spatial patterns and hierarchical visual features.

#     For Neural Architecture Search, useful architecture parameters
#     include the number of convolutional blocks, number of filters,
#     kernel sizes, activation functions, pooling operations,
#     normalization, dropout and attention.
#     """,

#     # Document 2: CNN architecture depth
#     """
#     Medical image classification commonly uses CNN-based
#     architectures because convolutional layers can extract
#     low-level, intermediate-level and high-level visual
#     representations.

#     A CNN search space can contain different numbers of
#     convolutional blocks.

#     Increasing the number of blocks can provide greater
#     representational depth, while a smaller number of blocks
#     can reduce computational cost.

#     Neural Architecture Search can therefore explore different
#     network depths rather than relying on one manually selected
#     architecture.
#     """,

#     # Document 3: Number of filters
#     """
#     The number of filters in convolutional layers controls
#     the number of feature maps learned by the network.

#     Typical CNN architecture search spaces can include
#     filter choices such as 32, 64, 128 and 256.

#     Smaller filter counts can reduce computational cost,
#     while larger filter counts can provide greater feature
#     representation capacity.

#     Filter count can therefore be treated as an architecture
#     search parameter for healthcare image classification.
#     """,

#     # Document 4: Kernel size
#     """
#     Kernel size is an important CNN architecture parameter.

#     Common convolutional kernel sizes include 3x3 and 5x5.

#     Smaller kernels can capture local spatial patterns,
#     while larger kernels can provide a wider receptive field.

#     For Neural Architecture Search, kernel size can be
#     included as a categorical search parameter.

#     A healthcare image classification NAS system can compare
#     architectures using different kernel sizes.
#     """,

#     # Document 5: Activation functions
#     """
#     Activation functions introduce non-linearity into
#     neural networks.

#     ReLU is widely used in convolutional neural networks
#     because of its computational simplicity.

#     GELU is another activation function that can be explored
#     as an alternative.

#     Activation function selection can therefore be included
#     in the neural architecture search space.
#     """,

#     # Document 6: Pooling
#     """
#     Pooling operations are commonly used in convolutional
#     neural networks to reduce spatial dimensions.

#     Max pooling selects strong local responses, while
#     average pooling computes the average activation
#     within a region.

#     Both max pooling and average pooling can be included
#     as architecture choices during Neural Architecture Search.

#     The NAS system can determine which configuration provides
#     better validation performance for the target healthcare
#     image classification task.
#     """,

#     # Document 7: Batch normalization
#     """
#     Batch normalization can be used in convolutional neural
#     networks to normalize intermediate activations.

#     It can improve training stability and can sometimes
#     accelerate convergence.

#     Whether batch normalization should be used can be treated
#     as a Boolean architecture decision during Neural
#     Architecture Search.

#     The LLM can therefore include batch normalization as
#     one of the searchable architecture parameters.
#     """,

#     # Document 8: Dropout
#     """
#     Dropout is a regularization technique that randomly
#     removes a fraction of activations during training.

#     Dropout can help reduce overfitting in neural networks.

#     Candidate dropout rates such as 0.2, 0.3, 0.4 and 0.5
#     can be included in a neural architecture search space.

#     The optimal value can then be determined through
#     experimentation on the target dataset.
#     """,

#     # Document 9: Attention
#     """
#     Attention mechanisms can be incorporated into CNN-based
#     medical image classification architectures.

#     Attention can allow a model to assign different importance
#     to feature representations.

#     For Neural Architecture Search, the presence or absence
#     of an attention module can be treated as a Boolean
#     architecture choice.

#     This allows NAS to compare architectures with and
#     without attention mechanisms.
#     """,

#     # Document 10: Residual connections
#     """
#     Residual connections are commonly used to construct
#     deeper convolutional neural networks.

#     Residual connections provide shortcut paths between layers
#     and can help information and gradients flow through
#     deeper networks.

#     Residual blocks can be considered as an architecture
#     option when designing search spaces for medical image
#     classification.

#     The LLM can use this knowledge when generating
#     candidate architecture configurations.
#     """,

#     # Document 11: Classification task
#     """
#     Chest X-ray image classification can involve binary or
#     multi-class classification depending on the dataset.

#     For a binary normal-versus-pneumonia task, the output
#     layer can contain a single classification output with
#     an appropriate binary classification loss.

#     The architecture search space should therefore be
#     conditioned on the machine-learning task specified
#     by the user.
#     """,

#     # Document 12: Image preprocessing
#     """
#     Healthcare image classification datasets can contain
#     variations in image resolution, acquisition conditions
#     and class distribution.

#     Image preprocessing can include resizing images to a
#     consistent input size and normalization of pixel values.

#     Data augmentation can also be considered during model
#     training to improve robustness.

#     These preprocessing decisions are related to the
#     machine-learning pipeline but should be kept separate
#     from the main neural architecture search parameters.
#     """,

#     # Document 13: Class imbalance
#     """
#     Class imbalance can affect the evaluation of healthcare
#     image classification models.

#     Accuracy alone may not provide a complete evaluation
#     when class distributions are unequal.

#     Precision, recall and F1 score can provide additional
#     information about classification performance.

#     Therefore, a healthcare NAS evaluation pipeline can use
#     multiple metrics rather than relying only on accuracy.
#     """,

#     # Document 14: Evaluation metrics
#     """
#     For binary healthcare image classification, useful
#     evaluation metrics include accuracy, precision, recall,
#     F1 score and area under the ROC curve.

#     The selected evaluation metric should reflect the
#     machine-learning objective and characteristics of
#     the dataset.

#     Multiple metrics can be used when comparing candidate
#     neural architectures generated by NAS.
#     """,

#     # Document 15: NAS search space
#     """
#     Neural Architecture Search can be represented as a
#     search over a predefined set of architecture decisions.

#     A search space can include network depth, filter count,
#     kernel size, activation function, pooling type,
#     normalization, dropout and attention.

#     An LLM can analyze a domain-specific machine-learning
#     problem and generate a domain-informed search space.

#     The generated search space can then be passed to a NAS
#     algorithm that creates and evaluates candidate
#     neural architectures.
#     """,

#     # Document 16: LLM-guided NAS
#     """
#     LLM-guided Neural Architecture Search differs from a
#     conventional manually designed architecture pipeline.

#     Instead of directly selecting a fixed CNN architecture,
#     the LLM analyzes the task description and relevant
#     domain knowledge and proposes architecture parameters
#     that can be explored.

#     Retrieval-Augmented Generation can provide
#     domain-specific context to the LLM before
#     architecture generation.

#     The resulting architecture search space should be
#     validated before being passed to the NAS engine.
#     """,

#     # Document 17: RAG
#     """
#     Retrieval-Augmented Generation consists of a retrieval
#     stage followed by a generation stage.

#     In the retrieval stage, the user's query and stored
#     knowledge documents are represented as numerical
#     embeddings.

#     Similarity between the query embedding and document
#     embeddings is calculated to identify relevant knowledge.

#     The retrieved documents are then included in the prompt
#     provided to the LLM.

#     This allows the LLM to use domain-specific information
#     when generating architecture guidance.
#     """,

#     # Document 18: Sentence Transformer embeddings
#     """
#     Sentence Transformer models can convert healthcare
#     machine-learning text into dense semantic vectors.

#     The all-MiniLM-L6-v2 model produces a 384-dimensional
#     embedding representation.

#     These embeddings allow semantically similar queries
#     and documents to be compared using vector similarity.

#     In the healthcare NAS system, the embedding model is
#     used for RAG retrieval and is separate from the Qwen
#     language model used for architecture generation.
#     """,

#     # Document 19: LLM inputs and output
#     """
#     The LLM component receives two important inputs:

#     1. The user's healthcare machine-learning problem.
#     2. Relevant healthcare knowledge retrieved by RAG.

#     These inputs are combined into a structured prompt.

#     Qwen then generates an architecture search specification
#     containing possible neural network design choices.

#     The generated output is represented as structured JSON
#     so that the downstream NAS component can process it.
#     """,

#     # Document 20: Validation
#     """
#     A validation layer is important when an LLM generates
#     neural architecture parameters.

#     The validator can check whether required parameters are
#     present and whether generated values belong to
#     allowed ranges.

#     For example, valid kernel sizes can be restricted to
#     3, 5 and 7, while valid activation functions can be
#     restricted to ReLU and GELU.

#     This prevents invalid LLM-generated configurations
#     from being passed directly to the NAS engine.
#     """,

#     # Document 21: Healthcare NAS search space
#     """
#     A healthcare-specific NAS search space can contain
#     parameters such as:

#     Number of blocks:
#         3, 4, 5, 6

#     Filters:
#         32, 64, 128, 256

#     Kernel sizes:
#         3, 5

#     Activation:
#         ReLU, GELU

#     Pooling:
#         Max pooling, average pooling

#     Batch normalization:
#         Enabled or disabled

#     Dropout:
#         0.2, 0.3, 0.4, 0.5

#     Attention:
#         Enabled or disabled

#     These parameters define possible CNN architectures
#     that can subsequently be explored by NAS.
#     """,

#     # Document 22: Architecture parameters vs hyperparameters
#     """
#     Training hyperparameters should be distinguished from
#     architecture parameters.

#     Architecture parameters describe the structure of the
#     neural network, including depth, filters, kernel size,
#     activation, pooling and attention.

#     Training hyperparameters include learning rate,
#     optimizer and number of training epochs.

#     A complete NAS system can optimize architecture choices
#     while separately configuring the training procedure.
#     """,

#     # Document 23: LLM-to-NAS interface
#     """
#     The LLM-to-NAS interface can be implemented using
#     structured JSON.

#     The LLM produces fields describing the healthcare task,
#     data type, problem type, architecture search space,
#     training configuration and evaluation metrics.

#     The validation layer checks the JSON before the NAS
#     engine consumes it.

#     This creates a clear separation between natural-language
#     reasoning and machine-readable architecture generation.
#     """,

#     # Document 24: Complete workflow
#     """
#     For a chest X-ray classification use case, the complete
#     LLM-guided architecture workflow can be represented as:

#     Healthcare problem
#         ->
#     Healthcare knowledge retrieval
#         ->
#     Semantic embeddings
#         ->
#     Relevant context
#         ->
#     LLM prompt
#         ->
#     Qwen architecture generation
#         ->
#     Structured NAS search space
#         ->
#     Validation
#         ->
#     NAS candidate generation
#         ->
#     Candidate evaluation
#         ->
#     Best architecture

#     The LLM therefore acts as a guide for defining the
#     architecture search space rather than replacing the
#     NAS optimization process.
#     """
# ]


# # Basic information about the knowledge base
# KNOWLEDGE_BASE_NAME = "Healthcare LLM-Guided NAS Knowledge Base"
# KNOWLEDGE_BASE_DOMAIN = "Healthcare"

# # Count the available knowledge documents
# KNOWLEDGE_BASE_DOCUMENT_COUNT = len(HEALTHCARE_KNOWLEDGE)


# # Return all healthcare knowledge
# def get_healthcare_knowledge():
#     return HEALTHCARE_KNOWLEDGE


# # Return the number of knowledge documents
# def get_document_count():
#     return len(HEALTHCARE_KNOWLEDGE)

# ============================================================
# Healthcare Knowledge Base for RAG
# ============================================================
#
# This knowledge base is intentionally NOT limited to
# chest X-ray / pneumonia.
#
# It contains general healthcare machine-learning knowledge
# plus examples covering different medical imaging modalities.
#
# The RAG system retrieves the most relevant documents based
# on the user's actual problem.
# ============================================================


HEALTHCARE_KNOWLEDGE = [

    # --------------------------------------------------------
    # Document 1: General healthcare image classification
    # --------------------------------------------------------
    """
    Healthcare medical-image classification can be formulated
    as a binary or multi-class classification problem depending
    on the dataset and target labels.

    CNN-based architectures are commonly suitable for image
    classification because convolutional operations can learn
    spatial patterns and hierarchical visual representations.

    Neural Architecture Search can explore architecture choices
    such as convolutional depth, filters, kernel size,
    activation functions, pooling, normalization, dropout,
    attention and residual connections.
    """,

    # --------------------------------------------------------
    # Document 2: Medical imaging modalities
    # --------------------------------------------------------
    """
    Healthcare machine-learning systems can use different
    imaging modalities including X-ray, MRI, CT, ultrasound,
    dermoscopy and retinal images.

    The architecture should be selected according to the
    characteristics of the input data and the machine-learning
    task.

    The LLM should identify the data modality from the user's
    problem rather than assuming a specific modality.
    """,

    # --------------------------------------------------------
    # Document 3: X-ray imaging
    # --------------------------------------------------------
    """
    X-ray images are two-dimensional radiographic images that
    can be used for healthcare image classification tasks.

    Example applications include detecting abnormalities,
    classifying disease categories and distinguishing normal
    from abnormal images.

    CNN architecture search can explore different numbers of
    convolutional blocks, filters, kernel sizes, pooling
    methods and regularization settings.
    """,

    # --------------------------------------------------------
    # Document 4: Brain MRI
    # --------------------------------------------------------
    """
    Brain MRI images can be used for machine-learning tasks
    involving brain abnormality and tumor classification.

    A brain MRI classification problem may be binary or
    multi-class depending on the dataset.

    CNN-based architectures can learn spatial patterns from
    MRI images.

    Neural Architecture Search can explore convolutional depth,
    filter count, kernel size, activation functions, pooling,
    normalization, dropout and attention mechanisms.

    The exact problem type should be determined from the user's
    stated target classes.
    """,

    # --------------------------------------------------------
    # Document 5: CT imaging
    # --------------------------------------------------------
    """
    CT images can be used for healthcare machine-learning
    classification and detection tasks.

    Depending on the dataset, CT data may be represented as
    individual two-dimensional slices or as three-dimensional
    volumes.

    Architecture selection should consider whether the input
    is treated as 2D image data or 3D volumetric data.
    """,

    # --------------------------------------------------------
    # Document 6: Skin / dermoscopy images
    # --------------------------------------------------------
    """
    Dermoscopy and skin images can be used for medical-image
    classification tasks.

    CNN architectures can extract visual features such as
    texture, shape and spatial patterns.

    NAS can explore different convolutional architectures,
    regularization settings and attention mechanisms for
    skin-image classification.
    """,

    # --------------------------------------------------------
    # Document 7: Retinal images
    # --------------------------------------------------------
    """
    Retinal fundus images can be used for healthcare image
    classification tasks.

    CNN-based models can learn hierarchical visual features
    from retinal images.

    Architecture search can investigate network depth,
    filter count, kernel size, pooling, normalization,
    dropout and attention.
    """,

    # --------------------------------------------------------
    # Document 8: CNN architecture depth
    # --------------------------------------------------------
    """
    CNN architecture depth describes the number of
    convolutional blocks or layers in the network.

    Increasing depth can increase representational capacity,
    while smaller networks can reduce computational cost.

    NAS can therefore search over multiple possible numbers
    of convolutional blocks instead of using one manually
    selected network depth.
    """,

    # --------------------------------------------------------
    # Document 9: Number of filters
    # --------------------------------------------------------
    """
    The number of convolutional filters controls the number
    of feature maps learned by a CNN.

    Candidate filter values can include 16, 32, 64, 128,
    256 and 512.

    Smaller filter counts generally reduce computational
    requirements, while larger values can provide greater
    feature representation capacity.

    Filter count can therefore be included as a NAS parameter.
    """,

    # --------------------------------------------------------
    # Document 10: Kernel size
    # --------------------------------------------------------
    """
    Kernel size determines the local spatial region processed
    by a convolution operation.

    Common CNN kernel sizes include 3x3, 5x5 and 7x7.

    Smaller kernels focus on local patterns while larger
    kernels provide a wider receptive field.

    Kernel size can therefore be treated as a categorical
    architecture search parameter.
    """,

    # --------------------------------------------------------
    # Document 11: Activation functions
    # --------------------------------------------------------
    """
    Activation functions introduce non-linearity into neural
    networks.

    ReLU is widely used in CNN architectures because of its
    computational simplicity.

    GELU can also be explored as an alternative activation
    function.

    Activation selection can be included in the NAS search
    space.
    """,

    # --------------------------------------------------------
    # Document 12: Pooling
    # --------------------------------------------------------
    """
    Pooling operations can reduce spatial dimensions in
    convolutional neural networks.

    Max pooling selects strong local responses, while average
    pooling computes an average activation over a region.

    Both max pooling and average pooling can be considered
    architecture choices during NAS.
    """,

    # --------------------------------------------------------
    # Document 13: Batch normalization
    # --------------------------------------------------------
    """
    Batch normalization can normalize intermediate neural
    network activations.

    It can improve training stability and may accelerate
    convergence.

    Whether batch normalization is enabled can be treated
    as an architecture choice during NAS.
    """,

    # --------------------------------------------------------
    # Document 14: Dropout
    # --------------------------------------------------------
    """
    Dropout is a regularization technique used to reduce
    overfitting.

    Candidate dropout rates can include 0.2, 0.3, 0.4 and
    0.5.

    Dropout rate can be treated as a searchable parameter
    during neural architecture experimentation.
    """,

    # --------------------------------------------------------
    # Document 15: Attention
    # --------------------------------------------------------
    """
    Attention mechanisms can be incorporated into CNN-based
    healthcare image architectures.

    Attention can allow a model to assign different importance
    to feature representations.

    The presence or absence of an attention module can be
    represented as a Boolean NAS architecture choice.
    """,

    # --------------------------------------------------------
    # Document 16: Residual connections
    # --------------------------------------------------------
    """
    Residual connections provide shortcut paths between
    neural network layers.

    They can help information and gradients flow through
    deeper networks.

    Residual blocks can therefore be included as an optional
    architecture component in a CNN search space.
    """,

    # --------------------------------------------------------
    # Document 17: Classification types
    # --------------------------------------------------------
    """
    Healthcare image classification can be binary or
    multi-class.

    Binary classification involves two target classes.

    Multi-class classification involves more than two target
    classes.

    The architecture specification should be conditioned on
    the classification problem described by the user and
    should not assume binary classification unless the user
    specifies two classes.
    """,

    # --------------------------------------------------------
    # Document 18: Detection and segmentation
    # --------------------------------------------------------
    """
    Healthcare computer-vision problems can include image
    classification, object detection and image segmentation.

    Classification predicts a class for an image or sample.

    Detection identifies objects or regions of interest.

    Segmentation assigns labels to individual pixels or
    regions.

    The LLM should identify the task from the user's problem
    before generating the architecture search space.
    """,

    # --------------------------------------------------------
    # Document 19: Preprocessing
    # --------------------------------------------------------
    """
    Medical-image preprocessing can include resizing,
    normalization and data augmentation.

    These preprocessing operations belong to the broader
    machine-learning pipeline and should be distinguished
    from the main neural architecture search parameters.
    """,

    # --------------------------------------------------------
    # Document 20: Class imbalance
    # --------------------------------------------------------
    """
    Healthcare datasets may contain unequal numbers of samples
    across classes.

    Accuracy alone may therefore be insufficient for evaluating
    classification models.

    Precision, recall, F1 score and ROC-AUC can provide
    additional information about model performance.

    Evaluation metrics should be selected according to the
    machine-learning task and dataset characteristics.
    """,

    # --------------------------------------------------------
    # Document 21: Evaluation metrics
    # --------------------------------------------------------
    """
    Common evaluation metrics for healthcare classification
    include accuracy, precision, recall, F1 score and ROC-AUC.

    Multiple metrics can be used when comparing candidate
    architectures generated by NAS.

    The evaluation metric should reflect the actual task
    rather than being assumed automatically.
    """,

    # --------------------------------------------------------
    # Document 22: NAS search space
    # --------------------------------------------------------
    """
    Neural Architecture Search represents architecture design
    as a search over multiple possible decisions.

    A CNN search space can include:

    - number of convolutional blocks
    - number of filters
    - kernel sizes
    - activation functions
    - pooling
    - batch normalization
    - dropout
    - attention
    - residual connections

    NAS can combine these choices to create and evaluate
    candidate neural network architectures.
    """,

    # --------------------------------------------------------
    # Document 23: LLM-guided NAS
    # --------------------------------------------------------
    """
    LLM-guided Neural Architecture Search uses a language model
    to guide the definition of an architecture search space.

    The LLM receives the user's machine-learning problem and
    relevant domain knowledge retrieved by RAG.

    Instead of selecting one fixed architecture, the LLM
    proposes multiple architecture choices that can be
    explored by a downstream NAS algorithm.
    """,

    # --------------------------------------------------------
    # Document 24: RAG
    # --------------------------------------------------------
    """
    Retrieval-Augmented Generation consists of retrieval
    followed by generation.

    The user's query and stored knowledge documents are
    converted into numerical embeddings.

    Similarity between the query embedding and document
    embeddings is calculated.

    Relevant documents are retrieved and inserted into the
    LLM prompt.

    The LLM then uses the retrieved context when generating
    its response.
    """,

    # --------------------------------------------------------
    # Document 25: Embeddings
    # --------------------------------------------------------
    """
    Sentence Transformer models can convert healthcare
    machine-learning text into dense numerical vectors.

    The all-MiniLM-L6-v2 model produces a 384-dimensional
    embedding representation.

    These embeddings allow semantically related queries and
    documents to be compared using vector similarity.

    The embedding model is separate from the Qwen language
    model used for architecture generation.
    """,

    # --------------------------------------------------------
    # Document 26: LLM-to-NAS interface
    # --------------------------------------------------------
    """
    The interface between the LLM and NAS can use structured
    JSON.

    The LLM can produce:

    - healthcare domain
    - machine-learning task
    - data type
    - problem type
    - architecture search space
    - training configuration
    - evaluation metrics

    The generated JSON should be validated before it is passed
    to the NAS engine.
    """,

    # --------------------------------------------------------
    # Document 27: Validation
    # --------------------------------------------------------
    """
    LLM-generated architecture parameters should be validated
    before being consumed by a NAS engine.

    Validation can check whether required fields are present
    and whether values belong to permitted ranges.

    Example valid kernel sizes include 3, 5 and 7.

    Example activation choices include ReLU and GELU.

    Validation prevents malformed or unsupported architecture
    configurations from being passed downstream.
    """,

    # --------------------------------------------------------
    # Document 28: Complete workflow
    # --------------------------------------------------------
    """
    The general LLM-guided healthcare NAS workflow is:

    Healthcare machine-learning problem
        ->
    RAG retrieval
        ->
    Semantic embeddings
        ->
    Relevant healthcare context
        ->
    LLM prompt
        ->
    Qwen architecture guidance
        ->
    Structured NAS search space
        ->
    Search-space validation
        ->
    NAS candidate generation
        ->
    Candidate training and evaluation
        ->
    Best architecture

    The LLM guides the architecture search but does not replace
    the NAS optimization process.
    """
]


# ============================================================
# Knowledge Base Information
# ============================================================

KNOWLEDGE_BASE_NAME = (
    "Healthcare LLM-Guided NAS Knowledge Base"
)

KNOWLEDGE_BASE_DOMAIN = "Healthcare"

KNOWLEDGE_BASE_DOCUMENT_COUNT = len(
    HEALTHCARE_KNOWLEDGE
)


def get_healthcare_knowledge():
    """Return all healthcare knowledge documents."""
    return HEALTHCARE_KNOWLEDGE


def get_document_count():
    """Return the number of knowledge documents."""
    return len(HEALTHCARE_KNOWLEDGE)