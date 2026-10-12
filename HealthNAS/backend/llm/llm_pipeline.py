# # import gc
# # import json

# # from .rag import HealthcareRAG
# # from .llm_model import HealthcareLLM


# # class HealthcareLLMPipeline:
# #     """
# #     Main pipeline connecting RAG, Qwen and NAS.

# #     Flow:
# #     Healthcare problem
# #         -> RAG retrieval
# #         -> Prompt construction
# #         -> Qwen
# #         -> JSON parsing
# #         -> Search-space validation
# #         -> NAS handoff
# #     """

# #     def __init__(self):
# #         print("\n")
# #         print("=" * 70)
# #         print("INITIALIZING HEALTHCARE LLM + RAG PIPELINE")
# #         print("=" * 70)

# #         # Initialize the healthcare RAG system
# #         self.rag = HealthcareRAG()

# #         # Qwen is loaded only when it is needed
# #         self.llm = None

# #         print("\nHealthcare RAG initialized successfully.")

# #     def build_prompt(self, query, retrieved_documents):
# #         """Build the prompt using the query and retrieved knowledge."""

# #         # Add retrieved documents to the prompt
# #         context_parts = []

# #         for i, doc in enumerate(retrieved_documents, start=1):
# #             context_parts.append(
# #                 f"""
# # Document {i}
# # Similarity Score: {doc['score']:.4f}

# # Healthcare Knowledge:
# # {doc['document'].strip()}
# # """
# #             )

# #         context = "\n".join(context_parts)

# #         # Combine the user problem with the retrieved context
# #         prompt = f"""
# # You are an AI system for:

# # LLM-Guided Neural Architecture Search
# # for Domain-Specific AI.

# # SELECTED DOMAIN:
# # Healthcare

# # USER MACHINE-LEARNING PROBLEM:

# # {query}

# # RETRIEVED HEALTHCARE KNOWLEDGE:

# # {context}

# # YOUR TASK

# # Analyze the healthcare machine-learning problem and
# # generate a neural architecture SEARCH SPACE that can
# # be passed to a Neural Architecture Search system.

# # The LLM must GUIDE the NAS system.

# # Do NOT simply recommend one fixed architecture
# # such as ResNet-50.

# # Instead, provide multiple architecture choices
# # that NAS can explore.

# # ARCHITECTURE PARAMETERS

# # The architecture search space should consider:

# # 1. Number of convolutional blocks
# # 2. Number of filters
# # 3. Kernel size
# # 4. Activation function
# # 5. Pooling method
# # 6. Batch normalization
# # 7. Dropout
# # 8. Attention mechanism

# # Training hyperparameters may also be provided
# # separately, but they are NOT the main architecture
# # search dimensions.

# # STRICT OUTPUT REQUIREMENT

# # Return ONLY valid JSON.

# # Do NOT use Markdown.

# # Do NOT use:
# # ```json

# # Do NOT write explanations outside the JSON.

# # Use EXACTLY this structure:

# # {{
# #     "domain": "healthcare",

# #     "task": "image_classification",

# #     "data_type": "chest_xray",

# #     "problem_type": "binary_classification",

# #     "search_space": {{
# #         "num_blocks": [3, 4, 5, 6],

# #         "filters": [32, 64, 128, 256],

# #         "kernel_sizes": [3, 5],

# #         "activation": [
# #             "relu",
# #             "gelu"
# #         ],

# #         "pooling": [
# #             "max",
# #             "average"
# #         ],

# #         "batch_normalization": [
# #             true,
# #             false
# #         ],

# #         "dropout": [
# #             0.2,
# #             0.3,
# #             0.4,
# #             0.5
# #         ],

# #         "attention": [
# #             true,
# #             false
# #         ]
# #     }},

# #     "training": {{
# #         "optimizer": "adam",

# #         "learning_rate": [
# #             0.0001,
# #             0.001
# #         ],

# #         "loss": "binary_cross_entropy"
# #     }},

# #     "evaluation_metrics": [
# #         "accuracy",
# #         "precision",
# #         "recall",
# #         "f1"
# #     ],

# #     "reason": "Brief technical reason for this search space"
# # }}

# # RULES

# # 1. Return valid JSON only.
# # 2. Do not select only one fixed architecture.
# # 3. Generate a searchable architecture space.
# # 4. Keep architecture choices realistic.
# # 5. Make the choices appropriate for the stated
# #    healthcare machine-learning problem.
# # 6. Do not provide medical diagnosis.
# # 7. Do not provide treatment advice.
# # 8. Focus only on machine-learning architecture.
# # """

# #         return prompt

# #     def extract_json(self, response):
# #         """Extract the JSON object from the Qwen response."""

# #         response = response.strip()

# #         # Remove Markdown code fences if Qwen adds them
# #         response = response.replace("```json", "")
# #         response = response.replace("```", "")
# #         response = response.strip()

# #         # Find the JSON object in the response
# #         start = response.find("{")
# #         end = response.rfind("}")

# #         if start == -1 or end == -1:
# #             raise ValueError(
# #                 "Qwen did not return a valid JSON object."
# #             )

# #         json_text = response[start:end + 1]

# #         try:
# #             return json.loads(json_text)

# #         except json.JSONDecodeError as error:
# #             raise ValueError(
# #                 f"Invalid JSON returned by Qwen: {error}"
# #             )

# #     def validate_search_space(self, architecture):
# #         """
# #         Validate the architecture search space generated by Qwen.
# #         """

# #         # Check whether the LLM generated a search space
# #         if "search_space" not in architecture:
# #             raise ValueError(
# #                 "LLM output does not contain search_space."
# #             )

# #         search_space = architecture["search_space"]

# #         # Required NAS parameters
# #         required_fields = [
# #             "num_blocks",
# #             "filters",
# #             "kernel_sizes",
# #             "activation",
# #             "pooling",
# #             "batch_normalization",
# #             "dropout",
# #             "attention"
# #         ]

# #         for field in required_fields:
# #             if field not in search_space:
# #                 raise ValueError(
# #                     f"Missing NAS parameter: {field}"
# #                 )

# #         # Validate number of convolutional blocks
# #         search_space["num_blocks"] = [
# #             int(value)
# #             for value in search_space["num_blocks"]
# #             if 2 <= int(value) <= 8
# #         ]

# #         # Validate filter values
# #         allowed_filters = [
# #             16,
# #             32,
# #             64,
# #             128,
# #             256,
# #             512
# #         ]

# #         search_space["filters"] = [
# #             int(value)
# #             for value in search_space["filters"]
# #             if int(value) in allowed_filters
# #         ]

# #         # Validate kernel sizes
# #         search_space["kernel_sizes"] = [
# #             int(value)
# #             for value in search_space["kernel_sizes"]
# #             if int(value) in [3, 5, 7]
# #         ]

# #         # Validate activation functions
# #         search_space["activation"] = [
# #             str(value).lower()
# #             for value in search_space["activation"]
# #             if str(value).lower() in [
# #                 "relu",
# #                 "gelu"
# #             ]
# #         ]

# #         # Validate pooling methods
# #         search_space["pooling"] = [
# #             str(value).lower()
# #             for value in search_space["pooling"]
# #             if str(value).lower() in [
# #                 "max",
# #                 "average"
# #             ]
# #         ]

# #         # Validate dropout values
# #         search_space["dropout"] = [
# #             float(value)
# #             for value in search_space["dropout"]
# #             if 0.0 <= float(value) <= 0.7
# #         ]

# #         # Validate batch normalization choices
# #         search_space["batch_normalization"] = [
# #             bool(value)
# #             for value in search_space["batch_normalization"]
# #         ]

# #         # Validate attention choices
# #         search_space["attention"] = [
# #             bool(value)
# #             for value in search_space["attention"]
# #         ]

# #         # Use default values if Qwen returns empty choices
# #         if not search_space["num_blocks"]:
# #             search_space["num_blocks"] = [3, 4, 5]

# #         if not search_space["filters"]:
# #             search_space["filters"] = [32, 64, 128]

# #         if not search_space["kernel_sizes"]:
# #             search_space["kernel_sizes"] = [3, 5]

# #         if not search_space["activation"]:
# #             search_space["activation"] = ["relu"]

# #         if not search_space["pooling"]:
# #             search_space["pooling"] = ["max"]

# #         if not search_space["dropout"]:
# #             search_space["dropout"] = [0.2, 0.3, 0.4]

# #         if not search_space["batch_normalization"]:
# #             search_space["batch_normalization"] = [
# #                 True,
# #                 False
# #             ]

# #         if not search_space["attention"]:
# #             search_space["attention"] = [
# #                 True,
# #                 False
# #             ]

# #         # Display the validated search space
# #         print("\n")
# #         print("=" * 70)
# #         print("NAS SEARCH SPACE VALIDATION REPORT")
# #         print("=" * 70)

# #         print(
# #             f"\n✓ Number of blocks: "
# #             f"{search_space['num_blocks']}"
# #         )

# #         print(
# #             f"✓ Filters: "
# #             f"{search_space['filters']}"
# #         )

# #         print(
# #             f"✓ Kernel sizes: "
# #             f"{search_space['kernel_sizes']}"
# #         )

# #         print(
# #             f"✓ Activation: "
# #             f"{search_space['activation']}"
# #         )

# #         print(
# #             f"✓ Pooling: "
# #             f"{search_space['pooling']}"
# #         )

# #         print(
# #             f"✓ Batch normalization: "
# #             f"{search_space['batch_normalization']}"
# #         )

# #         print(
# #             f"✓ Dropout: "
# #             f"{search_space['dropout']}"
# #         )

# #         print(
# #             f"✓ Attention: "
# #             f"{search_space['attention']}"
# #         )

# #         print("\n✓ LLM output passed NAS validation.")

# #         return architecture

# #     def release_rag(self):
# #         """Release RAG resources before loading Qwen."""

# #         print("\nReleasing RAG resources...")

# #         # Release the embedding model
# #         if hasattr(self.rag, "embedding_model"):
# #             del self.rag.embedding_model

# #         # Release the vector index
# #         if hasattr(self.rag, "index"):
# #             del self.rag.index

# #         # Release document embeddings
# #         if hasattr(self.rag, "document_embeddings"):
# #             del self.rag.document_embeddings

# #         gc.collect()

# #         print("RAG resources released.")

# #     def process(self, query):
# #         """Run the complete RAG + LLM pipeline."""

# #         # Step 1: Retrieve relevant healthcare documents
# #         print("\n")
# #         print("=" * 70)
# #         print("STEP 1: RAG RETRIEVAL")
# #         print("=" * 70)

# #         retrieved_documents = self.rag.retrieve(
# #             query,
# #             top_k=3
# #         )

# #         print(
# #             f"\nRetrieved "
# #             f"{len(retrieved_documents)} "
# #             f"healthcare documents."
# #         )

# #         # Step 2: Build the Qwen prompt
# #         print("\n")
# #         print("=" * 70)
# #         print("STEP 2: BUILDING LLM PROMPT")
# #         print("=" * 70)

# #         prompt = self.build_prompt(
# #             query,
# #             retrieved_documents
# #         )

# #         print("Prompt created successfully.")

# #         # Step 3: Release RAG memory before loading Qwen
# #         self.release_rag()

# #         # Step 4: Initialize Qwen
# #         print("\n")
# #         print("=" * 70)
# #         print("STEP 3: QWEN LLM")
# #         print("=" * 70)

# #         self.llm = HealthcareLLM()

# #         # Step 5: Generate the NAS search space
# #         print("\n")
# #         print("=" * 70)
# #         print("STEP 4: GENERATING NAS SEARCH SPACE")
# #         print("=" * 70)

# #         raw_response = self.llm.generate(prompt)

# #         print("\nRaw Qwen response:")
# #         print(raw_response)

# #         # Step 6: Convert the response into Python JSON
# #         print("\n")
# #         print("=" * 70)
# #         print("STEP 5: PARSING JSON")
# #         print("=" * 70)

# #         architecture = self.extract_json(raw_response)

# #         print("JSON parsed successfully.")

# #         # Step 7: Validate the generated search space
# #         print("\n")
# #         print("=" * 70)
# #         print("STEP 6: VALIDATING NAS SEARCH SPACE")
# #         print("=" * 70)

# #         architecture = self.validate_search_space(
# #             architecture
# #         )

# #         # Step 8: Show the LLM-to-NAS handoff
# #         print("\n")
# #         print("=" * 70)
# #         print("LLM → NAS HANDOFF")
# #         print("=" * 70)

# #         print("\nHealthcare Task:")

# #         print(
# #             architecture.get(
# #                 "task",
# #                 "Not specified"
# #             )
# #         )

# #         print("\nData Type:")

# #         print(
# #             architecture.get(
# #                 "data_type",
# #                 "Not specified"
# #             )
# #         )

# #         print("\nArchitecture Search Space:")
# #         print("-" * 50)

# #         for parameter, values in architecture[
# #             "search_space"
# #         ].items():

# #             print(
# #                 f"{parameter}: {values}"
# #             )

# #         print(
# #             "\n✓ LLM-generated search space "
# #             "is ready for NAS."
# #         )

# #         # Return the results to the application
# #         return {
# #             "query": query,

# #             "retrieved_documents":
# #                 retrieved_documents,

# #             "architecture":
# #                 architecture
# #         }

# #     def close(self):
# #         """Close the LLM pipeline and release memory."""

# #         print("\nClosing Healthcare LLM pipeline...")

# #         # Release Qwen
# #         if self.llm is not None:
# #             del self.llm
# #             self.llm = None

# #         # Release RAG
# #         if hasattr(self, "rag"):
# #             del self.rag

# #         gc.collect()

# #         print("Pipeline closed.")

# import gc
# import json

# from .rag import HealthcareRAG
# from .llm_model import HealthcareLLM


# class HealthcareLLMPipeline:
#     """
#     Main pipeline connecting:

#     User Problem
#         ->
#     RAG Retrieval
#         ->
#     Prompt Construction
#         ->
#     Qwen LLM
#         ->
#     JSON Parsing
#         ->
#     Search-Space Validation
#         ->
#     NAS Handoff

#     The LLM does NOT select one fixed architecture.
#     It generates a search space that NAS can explore.
#     """

#     def __init__(self):

#         print("\n")
#         print("=" * 70)
#         print("INITIALIZING HEALTHCARE LLM + RAG PIPELINE")
#         print("=" * 70)

#         # Initialize RAG
#         self.rag = HealthcareRAG()

#         # Qwen will be initialized later
#         self.llm = None

#         print("\nHealthcare RAG initialized successfully.")

#     # ========================================================
#     # PROMPT CONSTRUCTION
#     # ========================================================

#     def build_prompt(self, query, retrieved_documents):
#         """
#         Build a dynamic prompt for Qwen.

#         IMPORTANT:
#         Nothing here assumes chest X-ray,
#         pneumonia or binary classification.
#         """

#         context_parts = []

#         for i, doc in enumerate(
#             retrieved_documents,
#             start=1
#         ):

#             context_parts.append(
#                 f"""
# DOCUMENT {i}

# Similarity Score:
# {doc['score']:.4f}

# Healthcare Knowledge:
# {doc['document'].strip()}
# """
#             )

#         context = "\n".join(context_parts)

#         prompt = f"""
# You are an AI system for:

# LLM-Guided Neural Architecture Search
# for Domain-Specific AI.

# DOMAIN:
# Healthcare

# ==================================================
# USER MACHINE-LEARNING PROBLEM
# ==================================================

# {query}

# ==================================================
# RETRIEVED HEALTHCARE KNOWLEDGE
# ==================================================

# {context}

# ==================================================
# YOUR TASK
# ==================================================

# Analyze the user's actual machine-learning problem.

# Use the retrieved healthcare knowledge as supporting
# context.

# You must identify the machine-learning characteristics
# from the user's problem.

# IMPORTANT:

# Do NOT assume that the problem is chest X-ray.

# Do NOT assume pneumonia.

# Do NOT assume binary classification.

# Do NOT assume MRI.

# Do NOT assume any particular imaging modality unless
# it is stated or strongly supported by the user's query.

# The user's actual problem has priority.

# ==================================================
# TASK IDENTIFICATION
# ==================================================

# Determine:

# 1. The healthcare domain.
# 2. The machine-learning task.
# 3. The data type or modality.
# 4. The problem type.

# Possible machine-learning tasks include:

# - image_classification
# - object_detection
# - image_segmentation
# - tabular_classification
# - regression
# - other

# Possible data types include:

# - chest_xray
# - brain_mri
# - ct_scan
# - ultrasound
# - dermoscopy
# - retinal_image
# - medical_image
# - tabular_healthcare_data
# - other

# Possible classification types include:

# - binary_classification
# - multi_class_classification
# - multi_label_classification
# - not_applicable

# Only select values supported by the user's actual
# machine-learning problem.

# ==================================================
# LLM-GUIDED NAS
# ==================================================

# The LLM must GUIDE the NAS system.

# Do NOT return only one fixed architecture.

# For example, do NOT simply return:

# "Use ResNet-50."

# Instead, generate a SEARCH SPACE containing
# multiple architecture choices that a NAS algorithm
# can explore.

# For image-based CNN problems, consider:

# 1. Number of convolutional blocks
# 2. Number of filters
# 3. Kernel size
# 4. Activation function
# 5. Pooling method
# 6. Batch normalization
# 7. Dropout
# 8. Attention
# 9. Residual connections

# For tasks where a parameter is not relevant,
# use an appropriate empty list or explain why it
# is not applicable.

# ==================================================
# ARCHITECTURE SEARCH SPACE
# ==================================================

# Use realistic candidate values.

# Possible values include:

# Number of blocks:
# [3, 4, 5, 6]

# Filters:
# [32, 64, 128, 256]

# Kernel sizes:
# [3, 5]

# Activation:
# ["relu", "gelu"]

# Pooling:
# ["max", "average"]

# Batch normalization:
# [true, false]

# Dropout:
# [0.2, 0.3, 0.4, 0.5]

# Attention:
# [true, false]

# Residual connections:
# [true, false]

# ==================================================
# TRAINING CONFIGURATION
# ==================================================

# Training hyperparameters are separate from the
# architecture search space.

# Possible training configuration:

# Optimizer:
# "adam"

# Learning rate:
# [0.0001, 0.001]

# For binary classification:
# "binary_cross_entropy"

# For multi-class classification:
# "categorical_cross_entropy"

# If the problem type is not classification,
# select an appropriate loss or mark it as
# "task_dependent".

# ==================================================
# EVALUATION
# ==================================================

# For classification problems, possible metrics include:

# - accuracy
# - precision
# - recall
# - f1
# - roc_auc

# For other tasks, select metrics appropriate
# to the actual machine-learning task.

# Do NOT claim that any metric has been achieved.

# These are evaluation metrics that a future
# NAS/training system can use.

# ==================================================
# STRICT JSON OUTPUT
# ==================================================

# Return ONLY valid JSON.

# Do NOT use Markdown.

# Do NOT use:

# ```json

# Do NOT write explanations outside JSON.

# Use exactly this structure:

# {{
#     "domain": "healthcare",

#     "task": "machine_learning_task",

#     "data_type": "actual_data_type",

#     "problem_type": "actual_problem_type",

#     "search_space": {{
#         "num_blocks": [3, 4, 5, 6],

#         "filters": [32, 64, 128, 256],

#         "kernel_sizes": [3, 5],

#         "activation": [
#             "relu",
#             "gelu"
#         ],

#         "pooling": [
#             "max",
#             "average"
#         ],

#         "batch_normalization": [
#             true,
#             false
#         ],

#         "dropout": [
#             0.2,
#             0.3,
#             0.4,
#             0.5
#         ],

#         "attention": [
#             true,
#             false
#         ],

#         "residual_connections": [
#             true,
#             false
#         ]
#     }},

#     "training": {{
#         "optimizer": "adam",

#         "learning_rate": [
#             0.0001,
#             0.001
#         ],

#         "loss": "task_dependent"
#     }},

#     "evaluation_metrics": [],

#     "reason": "Brief technical reason for the selected search space"
# }}

# ==================================================
# RULES
# ==================================================

# 1. Return valid JSON only.

# 2. Do not select one fixed architecture.

# 3. Generate a searchable architecture space.

# 4. Keep architecture choices realistic.

# 5. Use the user's actual problem.

# 6. Do not assume chest X-ray.

# 7. Do not assume pneumonia.

# 8. Do not assume binary classification.

# 9. Do not provide medical diagnosis.

# 10. Do not provide treatment advice.

# 11. Focus only on machine-learning architecture.

# 12. Do not claim actual model performance.

# 13. The output will be passed to a NAS module.
# """

#         return prompt

#     # ========================================================
#     # JSON EXTRACTION
#     # ========================================================

#     def extract_json(self, response):
#         """
#         Extract and parse the JSON object generated by Qwen.
#         """

#         response = response.strip()

#         # Remove Markdown fences if the LLM adds them
#         response = response.replace(
#             "```json",
#             ""
#         )

#         response = response.replace(
#             "```",
#             ""
#         )

#         response = response.strip()

#         start = response.find("{")
#         end = response.rfind("}")

#         if start == -1 or end == -1:

#             raise ValueError(
#                 "Qwen did not return a valid JSON object."
#             )

#         json_text = response[
#             start:end + 1
#         ]

#         try:

#             return json.loads(json_text)

#         except json.JSONDecodeError as error:

#             raise ValueError(
#                 f"Invalid JSON returned by Qwen: {error}"
#             )

#     # ========================================================
#     # VALIDATION
#     # ========================================================

#     def validate_search_space(self, architecture):
#         """
#         Validate the architecture search space
#         generated by Qwen.
#         """

#         if "search_space" not in architecture:

#             raise ValueError(
#                 "LLM output does not contain search_space."
#             )

#         search_space = architecture[
#             "search_space"
#         ]

#         # Required architecture parameters
#         required_fields = [
#             "num_blocks",
#             "filters",
#             "kernel_sizes",
#             "activation",
#             "pooling",
#             "batch_normalization",
#             "dropout",
#             "attention",
#             "residual_connections"
#         ]

#         for field in required_fields:

#             if field not in search_space:

#                 raise ValueError(
#                     f"Missing NAS parameter: {field}"
#                 )

#         # ----------------------------------------------------
#         # Number of blocks
#         # ----------------------------------------------------

#         valid_blocks = []

#         for value in search_space[
#             "num_blocks"
#         ]:

#             try:

#                 value = int(value)

#                 if 2 <= value <= 8:
#                     valid_blocks.append(value)

#             except (ValueError, TypeError):
#                 pass

#         search_space[
#             "num_blocks"
#         ] = valid_blocks

#         # ----------------------------------------------------
#         # Filters
#         # ----------------------------------------------------

#         allowed_filters = [
#             16,
#             32,
#             64,
#             128,
#             256,
#             512
#         ]

#         valid_filters = []

#         for value in search_space[
#             "filters"
#         ]:

#             try:

#                 value = int(value)

#                 if value in allowed_filters:
#                     valid_filters.append(value)

#             except (ValueError, TypeError):
#                 pass

#         search_space[
#             "filters"
#         ] = valid_filters

#         # ----------------------------------------------------
#         # Kernel sizes
#         # ----------------------------------------------------

#         valid_kernels = []

#         for value in search_space[
#             "kernel_sizes"
#         ]:

#             try:

#                 value = int(value)

#                 if value in [3, 5, 7]:
#                     valid_kernels.append(value)

#             except (ValueError, TypeError):
#                 pass

#         search_space[
#             "kernel_sizes"
#         ] = valid_kernels

#         # ----------------------------------------------------
#         # Activation
#         # ----------------------------------------------------

#         valid_activations = []

#         for value in search_space[
#             "activation"
#         ]:

#             value = str(value).lower()

#             if value in [
#                 "relu",
#                 "gelu"
#             ]:

#                 valid_activations.append(value)

#         search_space[
#             "activation"
#         ] = valid_activations

#         # ----------------------------------------------------
#         # Pooling
#         # ----------------------------------------------------

#         valid_pooling = []

#         for value in search_space[
#             "pooling"
#         ]:

#             value = str(value).lower()

#             if value in [
#                 "max",
#                 "average"
#             ]:

#                 valid_pooling.append(value)

#         search_space[
#             "pooling"
#         ] = valid_pooling

#         # ----------------------------------------------------
#         # Dropout
#         # ----------------------------------------------------

#         valid_dropout = []

#         for value in search_space[
#             "dropout"
#         ]:

#             try:

#                 value = float(value)

#                 if 0.0 <= value <= 0.7:

#                     valid_dropout.append(value)

#             except (ValueError, TypeError):
#                 pass

#         search_space[
#             "dropout"
#         ] = valid_dropout

#         # ----------------------------------------------------
#         # Boolean parameters
#         # ----------------------------------------------------

#         search_space[
#             "batch_normalization"
#         ] = [
#             bool(value)
#             for value in search_space[
#                 "batch_normalization"
#             ]
#         ]

#         search_space[
#             "attention"
#         ] = [
#             bool(value)
#             for value in search_space[
#                 "attention"
#             ]
#         ]

#         search_space[
#             "residual_connections"
#         ] = [
#             bool(value)
#             for value in search_space[
#                 "residual_connections"
#             ]
#         ]

#         # ----------------------------------------------------
#         # Defaults if Qwen returns empty values
#         # ----------------------------------------------------

#         if not search_space[
#             "num_blocks"
#         ]:

#             search_space[
#                 "num_blocks"
#             ] = [3, 4, 5]

#         if not search_space[
#             "filters"
#         ]:

#             search_space[
#                 "filters"
#             ] = [32, 64, 128]

#         if not search_space[
#             "kernel_sizes"
#         ]:

#             search_space[
#                 "kernel_sizes"
#             ] = [3, 5]

#         if not search_space[
#             "activation"
#         ]:

#             search_space[
#                 "activation"
#             ] = ["relu"]

#         if not search_space[
#             "pooling"
#         ]:

#             search_space[
#                 "pooling"
#             ] = ["max"]

#         if not search_space[
#             "dropout"
#         ]:

#             search_space[
#                 "dropout"
#             ] = [0.2, 0.3, 0.4]

#         if not search_space[
#             "batch_normalization"
#         ]:

#             search_space[
#                 "batch_normalization"
#             ] = [True, False]

#         if not search_space[
#             "attention"
#         ]:

#             search_space[
#                 "attention"
#             ] = [True, False]

#         if not search_space[
#             "residual_connections"
#         ]:

#             search_space[
#                 "residual_connections"
#             ] = [True, False]

#         # ----------------------------------------------------
#         # Display validation report
#         # ----------------------------------------------------

#         print("\n")
#         print("=" * 70)
#         print("NAS SEARCH SPACE VALIDATION REPORT")
#         print("=" * 70)

#         print(
#             "\n✓ Number of blocks:",
#             search_space["num_blocks"]
#         )

#         print(
#             "✓ Filters:",
#             search_space["filters"]
#         )

#         print(
#             "✓ Kernel sizes:",
#             search_space["kernel_sizes"]
#         )

#         print(
#             "✓ Activation:",
#             search_space["activation"]
#         )

#         print(
#             "✓ Pooling:",
#             search_space["pooling"]
#         )

#         print(
#             "✓ Batch normalization:",
#             search_space[
#                 "batch_normalization"
#             ]
#         )

#         print(
#             "✓ Dropout:",
#             search_space["dropout"]
#         )

#         print(
#             "✓ Attention:",
#             search_space["attention"]
#         )

#         print(
#             "✓ Residual connections:",
#             search_space[
#                 "residual_connections"
#             ]
#         )

#         print(
#             "\n✓ LLM output passed NAS validation."
#         )

#         return architecture

#     # ========================================================
#     # RELEASE RAG RESOURCES
#     # ========================================================

#     def release_rag(self):
#         """
#         Release RAG resources before loading Qwen.
#         This can reduce memory usage.
#         """

#         print("\nReleasing RAG resources...")

#         if hasattr(
#             self.rag,
#             "embedding_model"
#         ):

#             del self.rag.embedding_model

#         if hasattr(
#             self.rag,
#             "index"
#         ):

#             del self.rag.index

#         if hasattr(
#             self.rag,
#             "document_embeddings"
#         ):

#             del self.rag.document_embeddings

#         gc.collect()

#         print("RAG resources released.")

#     # ========================================================
#     # MAIN PIPELINE
#     # ========================================================

#     def process(self, query):
#         """
#         Run:

#         Query
#         ->
#         RAG
#         ->
#         Prompt
#         ->
#         Qwen
#         ->
#         JSON
#         ->
#         Validation
#         ->
#         NAS handoff
#         """

#         # ----------------------------------------------------
#         # STEP 1: RAG
#         # ----------------------------------------------------

#         print("\n")
#         print("=" * 70)
#         print("STEP 1: RAG RETRIEVAL")
#         print("=" * 70)

#         retrieved_documents = self.rag.retrieve(
#             query,
#             top_k=3
#         )

#         print(
#             f"\nRetrieved "
#             f"{len(retrieved_documents)} "
#             f"healthcare documents."
#         )

#         # ----------------------------------------------------
#         # STEP 2: Prompt
#         # ----------------------------------------------------

#         print("\n")
#         print("=" * 70)
#         print("STEP 2: BUILDING LLM PROMPT")
#         print("=" * 70)

#         prompt = self.build_prompt(
#             query,
#             retrieved_documents
#         )

#         print(
#             "Prompt created successfully."
#         )

#         # ----------------------------------------------------
#         # STEP 3: Release RAG memory
#         # ----------------------------------------------------

#         self.release_rag()

#         # ----------------------------------------------------
#         # STEP 4: Initialize Qwen
#         # ----------------------------------------------------

#         print("\n")
#         print("=" * 70)
#         print("STEP 3: QWEN LLM")
#         print("=" * 70)

#         self.llm = HealthcareLLM()

#         # ----------------------------------------------------
#         # STEP 5: Generate search space
#         # ----------------------------------------------------

#         print("\n")
#         print("=" * 70)
#         print("STEP 4: GENERATING NAS SEARCH SPACE")
#         print("=" * 70)

#         raw_response = self.llm.generate(
#             prompt
#         )

#         print("\nRaw Qwen response:")
#         print(raw_response)

#         # ----------------------------------------------------
#         # STEP 6: Parse JSON
#         # ----------------------------------------------------

#         print("\n")
#         print("=" * 70)
#         print("STEP 5: PARSING JSON")
#         print("=" * 70)

#         architecture = self.extract_json(
#             raw_response
#         )

#         print(
#             "JSON parsed successfully."
#         )

#         # ----------------------------------------------------
#         # STEP 7: Validate
#         # ----------------------------------------------------

#         print("\n")
#         print("=" * 70)
#         print("STEP 6: VALIDATING NAS SEARCH SPACE")
#         print("=" * 70)

#         architecture = self.validate_search_space(
#             architecture
#         )

#         # ----------------------------------------------------
#         # STEP 8: NAS handoff
#         # ----------------------------------------------------

#         print("\n")
#         print("=" * 70)
#         print("LLM → NAS HANDOFF")
#         print("=" * 70)

#         print(
#             "\nHealthcare Domain:"
#         )

#         print(
#             architecture.get(
#                 "domain",
#                 "Not specified"
#             )
#         )

#         print(
#             "\nMachine Learning Task:"
#         )

#         print(
#             architecture.get(
#                 "task",
#                 "Not specified"
#             )
#         )

#         print(
#             "\nData Type:"
#         )

#         print(
#             architecture.get(
#                 "data_type",
#                 "Not specified"
#             )
#         )

#         print(
#             "\nProblem Type:"
#         )

#         print(
#             architecture.get(
#                 "problem_type",
#                 "Not specified"
#             )
#         )

#         print(
#             "\nArchitecture Search Space:"
#         )

#         print("-" * 50)

#         for parameter, values in architecture[
#             "search_space"
#         ].items():

#             print(
#                 f"{parameter}: {values}"
#             )

#         print(
#             "\n✓ LLM-generated search space "
#             "is ready for NAS."
#         )

#         # ----------------------------------------------------
#         # RETURN RESULTS
#         # ----------------------------------------------------

#         return {
#             "query": query,

#             "retrieved_documents":
#                 retrieved_documents,

#             "architecture":
#                 architecture
#         }

#     # ========================================================
#     # CLOSE
#     # ========================================================

#     def close(self):
#         """
#         Release LLM and RAG resources.
#         """

#         print(
#             "\nClosing Healthcare LLM pipeline..."
#         )

#         if self.llm is not None:

#             del self.llm

#             self.llm = None

#         if hasattr(
#             self,
#             "rag"
#         ):

#             del self.rag

#         gc.collect()

#         print(
#             "Pipeline closed."
#         )
import gc
import json

from .rag import HealthcareRAG
from .llm_model import HealthcareLLM


class HealthcareLLMPipeline:
    """
    Main pipeline connecting:

    User Problem
        ->
    RAG Retrieval
        ->
    Prompt Construction
        ->
    Qwen LLM
        ->
    JSON Parsing
        ->
    Search-Space Validation
        ->
    NAS Handoff

    The LLM does NOT select one fixed architecture.
    It generates a search space that NAS can explore.
    """

    def __init__(self):

        print("\n")
        print("=" * 70)
        print("INITIALIZING HEALTHCARE LLM + RAG PIPELINE")
        print("=" * 70)

        # Initialize healthcare RAG
        self.rag = HealthcareRAG()

        # Qwen is initialized only when needed
        self.llm = None

        print("\nHealthcare RAG initialized successfully.")

    # ========================================================
    # PROMPT CONSTRUCTION
    # ========================================================

    def build_prompt(self, query, retrieved_documents):
        """
        Build a dynamic prompt for Qwen.

        The user's problem determines:
        - ML task
        - healthcare data modality
        - problem type
        - loss function
        - evaluation metrics
        - NAS search space
        """

        context_parts = []

        for i, doc in enumerate(
            retrieved_documents,
            start=1
        ):

            context_parts.append(
                f"""
DOCUMENT {i}

Similarity Score:
{doc['score']:.4f}

Healthcare Knowledge:
{doc['document'].strip()}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are an AI system for:

LLM-Guided Neural Architecture Search
for Domain-Specific AI.

DOMAIN:
Healthcare

==================================================
USER MACHINE-LEARNING PROBLEM
==================================================

{query}

==================================================
RETRIEVED HEALTHCARE KNOWLEDGE
==================================================

{context}

==================================================
YOUR TASK
==================================================

Analyze the USER MACHINE-LEARNING PROBLEM.

Use the retrieved healthcare knowledge only as
supporting context.

The user's actual problem has priority over the
retrieved documents.

You must determine the actual:

1. Machine-learning task
2. Healthcare data type / modality
3. Problem type
4. Appropriate architecture search space
5. Training loss
6. Evaluation metrics

DO NOT simply copy words from an output template.

==================================================
TASK IDENTIFICATION
==================================================

Choose the machine-learning task that best matches
the user's problem.

Allowed task values include:

- image_classification
- object_detection
- image_segmentation
- tabular_classification
- regression
- other

Examples:

If the user wants to determine whether an entire
medical image belongs to one class or another,
use:

"image_classification"

If the user wants bounding boxes around abnormalities,
use:

"object_detection"

If the user wants pixel-level regions or masks,
use:

"image_segmentation"

==================================================
DATA TYPE / MODALITY IDENTIFICATION
==================================================

Infer the data modality from the user's problem.

Allowed data_type values include:

- chest_xray
- brain_mri
- ct_scan
- ultrasound
- dermoscopy
- retinal_image
- medical_image
- tabular_healthcare_data
- other

Examples:

"brain MRI tumor classification"
-> brain_mri

"pneumonia using chest X-rays"
-> chest_xray

"lung abnormality using CT"
-> ct_scan

"skin lesion using dermoscopy"
-> dermoscopy

Do NOT assume a modality that the user did not
state or strongly imply.

==================================================
PROBLEM TYPE IDENTIFICATION
==================================================

Allowed problem_type values include:

- binary_classification
- multi_class_classification
- multi_label_classification
- regression
- detection
- segmentation
- not_applicable

Determine the type from the user's target.

For example:

"tumor or normal"
contains two mutually exclusive classes.

Therefore:

problem_type = "binary_classification"

If there are more than two mutually exclusive
classes:

problem_type = "multi_class_classification"

For object detection:

problem_type = "detection"

For image segmentation:

problem_type = "segmentation"

==================================================
IMPORTANT INFERENCE RULE
==================================================

You MUST infer:

"task"
"data_type"
"problem_type"

from the USER MACHINE-LEARNING PROBLEM.

NEVER return literal placeholder text such as:

"machine_learning_task"
"actual_data_type"
"actual_problem_type"

NEVER leave task, data_type or problem_type empty.

Example:

User problem:

"brain MRI detection of brain tumor or normal"

This means classification of the whole MRI into
two classes.

Correct interpretation:

task = "image_classification"
data_type = "brain_mri"
problem_type = "binary_classification"

The word "detection" alone does NOT necessarily
mean object detection.

Use "object_detection" only when the problem
requires locating objects, for example using
bounding boxes.

==================================================
LLM-GUIDED NEURAL ARCHITECTURE SEARCH
==================================================

The purpose of your output is to GUIDE a
Neural Architecture Search system.

Do NOT return only one fixed architecture.

For example, do NOT simply say:

"Use ResNet-50."

Instead, return multiple candidate architecture
values that a NAS algorithm can explore.

For image-based CNN problems, consider:

1. Number of convolutional blocks
2. Number of filters
3. Kernel sizes
4. Activation functions
5. Pooling methods
6. Batch normalization
7. Dropout
8. Attention
9. Residual connections

==================================================
ARCHITECTURE SEARCH SPACE
==================================================

Generate realistic candidate values.

Possible candidate values include:

Number of convolutional blocks:

[3, 4, 5, 6]

Filters:

[32, 64, 128, 256]

Kernel sizes:

[3, 5]

Activation functions:

["relu", "gelu"]

Pooling:

["max", "average"]

Batch normalization:

[true, false]

Dropout:

[0.2, 0.3, 0.4, 0.5]

Attention:

[true, false]

Residual connections:

[true, false]

You may select an appropriate subset of these
values based on the user's actual problem.

The search space must contain MULTIPLE choices.

==================================================
TRAINING CONFIGURATION
==================================================

Training hyperparameters are separate from the
architecture search space.

Optimizer:

"adam"

Possible learning rates:

[0.0001, 0.001]

Choose the loss function according to the
identified problem.

For binary classification:

"binary_cross_entropy"

For multi-class classification:

"categorical_cross_entropy"

For regression:

choose an appropriate regression loss.

For object detection or segmentation:

use a suitable task-dependent loss description.

Do NOT return "task_dependent" when the actual
loss can clearly be inferred from the problem.

==================================================
EVALUATION METRICS
==================================================

Choose metrics appropriate for the actual task.

For classification, possible metrics include:

- accuracy
- precision
- recall
- f1
- roc_auc

For regression, possible metrics include:

- mae
- mse
- rmse

For segmentation, possible metrics include:

- dice
- iou

For object detection, possible metrics include:

- map
- precision
- recall

Do NOT claim any metric value has already been
achieved.

These are only metrics that a future training/NAS
system can use.

For an identifiable classification problem,
evaluation_metrics MUST NOT be empty.

==================================================
STRICT JSON OUTPUT
==================================================

Return ONLY valid JSON.

Do NOT use Markdown.

Do NOT use ```json.

Do NOT provide explanations outside the JSON.

Return exactly these top-level fields:

domain
task
data_type
problem_type
search_space
training
evaluation_metrics
reason

Use this JSON structure.

IMPORTANT:

The empty strings below indicate fields that YOU
MUST fill using your analysis.

Do NOT leave them empty.

{{
    "domain": "healthcare",

    "task": "",

    "data_type": "",

    "problem_type": "",

    "search_space": {{
        "num_blocks": [
            3,
            4,
            5,
            6
        ],

        "filters": [
            32,
            64,
            128,
            256
        ],

        "kernel_sizes": [
            3,
            5
        ],

        "activation": [
            "relu",
            "gelu"
        ],

        "pooling": [
            "max",
            "average"
        ],

        "batch_normalization": [
            true,
            false
        ],

        "dropout": [
            0.2,
            0.3,
            0.4,
            0.5
        ],

        "attention": [
            true,
            false
        ],

        "residual_connections": [
            true,
            false
        ]
    }},

    "training": {{
        "optimizer": "adam",

        "learning_rate": [
            0.0001,
            0.001
        ],

        "loss": ""
    }},

    "evaluation_metrics": [],

    "reason": ""
}}

==================================================
FINAL RULES
==================================================

1. Return valid JSON only.

2. Generate a SEARCH SPACE, not one fixed model.

3. Use the user's actual healthcare problem.

4. Infer task from the user's problem.

5. Infer data_type from the user's problem.

6. Infer problem_type from the user's problem.

7. Infer the training loss from the problem type.

8. Populate evaluation_metrics with appropriate metrics.

9. Do NOT output:
   "machine_learning_task"

10. Do NOT output:
    "actual_data_type"

11. Do NOT output:
    "actual_problem_type"

12. Do NOT leave task empty.

13. Do NOT leave data_type empty.

14. Do NOT leave problem_type empty.

15. Do NOT leave loss empty when it can be inferred.

16. Do NOT assume chest X-ray.

17. Do NOT assume pneumonia.

18. Do NOT assume MRI unless supported by the user's query.

19. Do NOT automatically assume binary classification.

20. Do NOT provide medical diagnosis.

21. Do NOT provide treatment advice.

22. Focus only on machine-learning architecture.

23. Do NOT claim actual model performance.

24. The generated search space will be passed to NAS.
"""

        return prompt

    # ========================================================
    # JSON EXTRACTION
    # ========================================================

    def extract_json(self, response):
        """
        Extract and parse JSON generated by Qwen.
        """

        response = response.strip()

        # Remove Markdown code fences if Qwen adds them
        response = response.replace("```json", "")
        response = response.replace("```", "")
        response = response.strip()

        # Find JSON boundaries
        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end == -1:

            raise ValueError(
                "Qwen did not return a valid JSON object."
            )

        json_text = response[start:end + 1]

        try:

            return json.loads(json_text)

        except json.JSONDecodeError as error:

            raise ValueError(
                f"Invalid JSON returned by Qwen: {error}"
            )

    # ========================================================
    # METADATA VALIDATION
    # ========================================================

    def validate_metadata(self, architecture):
        """
        Validate task, data type and problem type.

        This prevents placeholder values from being
        silently passed to the NAS system.
        """

        required_metadata = [
            "domain",
            "task",
            "data_type",
            "problem_type"
        ]

        for field in required_metadata:

            if field not in architecture:

                raise ValueError(
                    f"LLM output is missing required field: {field}"
                )

        invalid_values = {
            "",
            "machine_learning_task",
            "actual_data_type",
            "actual_problem_type"
        }

        for field in [
            "task",
            "data_type",
            "problem_type"
        ]:

            value = str(
                architecture.get(field, "")
            ).strip().lower()

            if value in invalid_values:

                raise ValueError(
                    f"Qwen failed to infer '{field}'. "
                    f"Received invalid value: "
                    f"{architecture.get(field)}"
                )

        return architecture

    # ========================================================
    # SEARCH SPACE VALIDATION
    # ========================================================

    def validate_search_space(self, architecture):
        """
        Validate the architecture search space
        generated by Qwen.
        """

        # First validate metadata
        architecture = self.validate_metadata(
            architecture
        )

        if "search_space" not in architecture:

            raise ValueError(
                "LLM output does not contain search_space."
            )

        search_space = architecture[
            "search_space"
        ]

        # ----------------------------------------------------
        # Required architecture parameters
        # ----------------------------------------------------

        required_fields = [
            "num_blocks",
            "filters",
            "kernel_sizes",
            "activation",
            "pooling",
            "batch_normalization",
            "dropout",
            "attention",
            "residual_connections"
        ]

        for field in required_fields:

            if field not in search_space:

                raise ValueError(
                    f"Missing NAS parameter: {field}"
                )

        # ----------------------------------------------------
        # Number of blocks
        # ----------------------------------------------------

        valid_blocks = []

        for value in search_space["num_blocks"]:

            try:

                value = int(value)

                if 2 <= value <= 8:
                    valid_blocks.append(value)

            except (ValueError, TypeError):
                pass

        search_space["num_blocks"] = valid_blocks

        # ----------------------------------------------------
        # Filters
        # ----------------------------------------------------

        allowed_filters = [
            16,
            32,
            64,
            128,
            256,
            512
        ]

        valid_filters = []

        for value in search_space["filters"]:

            try:

                value = int(value)

                if value in allowed_filters:
                    valid_filters.append(value)

            except (ValueError, TypeError):
                pass

        search_space["filters"] = valid_filters

        # ----------------------------------------------------
        # Kernel sizes
        # ----------------------------------------------------

        valid_kernels = []

        for value in search_space["kernel_sizes"]:

            try:

                value = int(value)

                if value in [3, 5, 7]:
                    valid_kernels.append(value)

            except (ValueError, TypeError):
                pass

        search_space["kernel_sizes"] = valid_kernels

        # ----------------------------------------------------
        # Activations
        # ----------------------------------------------------

        valid_activations = []

        for value in search_space["activation"]:

            value = str(value).lower()

            if value in [
                "relu",
                "gelu"
            ]:

                valid_activations.append(value)

        search_space["activation"] = valid_activations

        # ----------------------------------------------------
        # Pooling
        # ----------------------------------------------------

        valid_pooling = []

        for value in search_space["pooling"]:

            value = str(value).lower()

            if value in [
                "max",
                "average"
            ]:

                valid_pooling.append(value)

        search_space["pooling"] = valid_pooling

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        valid_dropout = []

        for value in search_space["dropout"]:

            try:

                value = float(value)

                if 0.0 <= value <= 0.7:
                    valid_dropout.append(value)

            except (ValueError, TypeError):
                pass

        search_space["dropout"] = valid_dropout

        # ----------------------------------------------------
        # Boolean parameters
        # ----------------------------------------------------

        search_space["batch_normalization"] = (
            self._validate_boolean_list(
                search_space["batch_normalization"]
            )
        )

        search_space["attention"] = (
            self._validate_boolean_list(
                search_space["attention"]
            )
        )

        search_space["residual_connections"] = (
            self._validate_boolean_list(
                search_space["residual_connections"]
            )
        )

        # ----------------------------------------------------
        # Defaults if Qwen returns invalid/empty values
        # ----------------------------------------------------

        if not search_space["num_blocks"]:
            search_space["num_blocks"] = [
                3,
                4,
                5
            ]

        if not search_space["filters"]:
            search_space["filters"] = [
                32,
                64,
                128
            ]

        if not search_space["kernel_sizes"]:
            search_space["kernel_sizes"] = [
                3,
                5
            ]

        if not search_space["activation"]:
            search_space["activation"] = [
                "relu"
            ]

        if not search_space["pooling"]:
            search_space["pooling"] = [
                "max"
            ]

        if not search_space["dropout"]:
            search_space["dropout"] = [
                0.2,
                0.3,
                0.4
            ]

        if not search_space["batch_normalization"]:
            search_space["batch_normalization"] = [
                True,
                False
            ]

        if not search_space["attention"]:
            search_space["attention"] = [
                True,
                False
            ]

        if not search_space["residual_connections"]:
            search_space["residual_connections"] = [
                True,
                False
            ]

        # ----------------------------------------------------
        # Training validation
        # ----------------------------------------------------

        if "training" not in architecture:
            architecture["training"] = {
                "optimizer": "adam",
                "learning_rate": [
                    0.0001,
                    0.001
                ],
                "loss": "task_dependent"
            }

        # ----------------------------------------------------
        # Evaluation metrics validation
        # ----------------------------------------------------

        if "evaluation_metrics" not in architecture:
            architecture["evaluation_metrics"] = []

        # ----------------------------------------------------
        # Validation report
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("NAS SEARCH SPACE VALIDATION REPORT")
        print("=" * 70)

        print(
            "\n✓ Number of blocks:",
            search_space["num_blocks"]
        )

        print(
            "✓ Filters:",
            search_space["filters"]
        )

        print(
            "✓ Kernel sizes:",
            search_space["kernel_sizes"]
        )

        print(
            "✓ Activation:",
            search_space["activation"]
        )

        print(
            "✓ Pooling:",
            search_space["pooling"]
        )

        print(
            "✓ Batch normalization:",
            search_space["batch_normalization"]
        )

        print(
            "✓ Dropout:",
            search_space["dropout"]
        )

        print(
            "✓ Attention:",
            search_space["attention"]
        )

        print(
            "✓ Residual connections:",
            search_space["residual_connections"]
        )

        print(
            "\n✓ LLM output passed NAS validation."
        )

        return architecture

    # ========================================================
    # BOOLEAN VALIDATION
    # ========================================================

    def _validate_boolean_list(self, values):
        """
        Safely convert boolean-like values.

        Avoids the Python problem where:
        bool("false") == True
        """

        result = []

        for value in values:

            if isinstance(value, bool):

                result.append(value)

            elif isinstance(value, str):

                cleaned = value.strip().lower()

                if cleaned == "true":
                    result.append(True)

                elif cleaned == "false":
                    result.append(False)

        return result

    # ========================================================
    # RELEASE RAG RESOURCES
    # ========================================================

    def release_rag(self):
        """
        Release large RAG resources before using Qwen.

        This reduces memory usage.
        """

        print("\nReleasing RAG resources...")

        if hasattr(
            self.rag,
            "embedding_model"
        ):

            del self.rag.embedding_model

        if hasattr(
            self.rag,
            "index"
        ):

            del self.rag.index

        if hasattr(
            self.rag,
            "document_embeddings"
        ):

            del self.rag.document_embeddings

        gc.collect()

        print("RAG resources released.")

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def process(self, query):
        """
        Run complete pipeline:

        Query
        ->
        RAG Retrieval
        ->
        Prompt
        ->
        Qwen
        ->
        JSON
        ->
        Validation
        ->
        NAS Handoff
        """

        # ----------------------------------------------------
        # STEP 1: RAG RETRIEVAL
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("STEP 1: RAG RETRIEVAL")
        print("=" * 70)

        retrieved_documents = self.rag.retrieve(
            query,
            top_k=3
        )

        print(
            f"\nRetrieved "
            f"{len(retrieved_documents)} "
            f"healthcare documents."
        )

        # ----------------------------------------------------
        # STEP 2: BUILD PROMPT
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("STEP 2: BUILDING LLM PROMPT")
        print("=" * 70)

        prompt = self.build_prompt(
            query,
            retrieved_documents
        )

        print(
            "Prompt created successfully."
        )

        # ----------------------------------------------------
        # STEP 3: RELEASE RAG MEMORY
        # ----------------------------------------------------

        self.release_rag()

        # ----------------------------------------------------
        # STEP 4: INITIALIZE QWEN
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("STEP 3: QWEN LLM")
        print("=" * 70)

        self.llm = HealthcareLLM()

        # ----------------------------------------------------
        # STEP 5: GENERATE SEARCH SPACE
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("STEP 4: GENERATING NAS SEARCH SPACE")
        print("=" * 70)

        raw_response = self.llm.generate(
            prompt
        )

        print("\nRaw Qwen response:")
        print(raw_response)

        # ----------------------------------------------------
        # STEP 6: PARSE JSON
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("STEP 5: PARSING JSON")
        print("=" * 70)

        architecture = self.extract_json(
            raw_response
        )

        print(
            "JSON parsed successfully."
        )

        # ----------------------------------------------------
        # STEP 7: VALIDATE
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("STEP 6: VALIDATING NAS SEARCH SPACE")
        print("=" * 70)

        architecture = self.validate_search_space(
            architecture
        )

        # ----------------------------------------------------
        # STEP 8: NAS HANDOFF
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("LLM → NAS HANDOFF")
        print("=" * 70)

        print(
            "\nHealthcare Domain:"
        )

        print(
            architecture.get(
                "domain",
                "Not specified"
            )
        )

        print(
            "\nMachine Learning Task:"
        )

        print(
            architecture.get(
                "task",
                "Not specified"
            )
        )

        print(
            "\nData Type:"
        )

        print(
            architecture.get(
                "data_type",
                "Not specified"
            )
        )

        print(
            "\nProblem Type:"
        )

        print(
            architecture.get(
                "problem_type",
                "Not specified"
            )
        )

        print(
            "\nArchitecture Search Space:"
        )

        print("-" * 50)

        for parameter, values in architecture[
            "search_space"
        ].items():

            print(
                f"{parameter}: {values}"
            )

        print(
            "\nTraining Configuration:"
        )

        print("-" * 50)

        for parameter, values in architecture.get(
            "training",
            {}
        ).items():

            print(
                f"{parameter}: {values}"
            )

        print(
            "\nEvaluation Metrics:"
        )

        print(
            architecture.get(
                "evaluation_metrics",
                []
            )
        )

        print(
            "\nReason:"
        )

        print(
            architecture.get(
                "reason",
                "Not specified"
            )
        )

        print(
            "\n✓ LLM-generated search space "
            "is ready for NAS."
        )

        # ----------------------------------------------------
        # RETURN RESULTS
        # ----------------------------------------------------

        return {
            "query": query,

            "retrieved_documents":
                retrieved_documents,

            "architecture":
                architecture
        }

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        """
        Release LLM and RAG resources.
        """

        print(
            "\nClosing Healthcare LLM pipeline..."
        )

        if self.llm is not None:

            del self.llm
            self.llm = None

        if hasattr(
            self,
            "rag"
        ):

            del self.rag

        gc.collect()

        print(
            "Pipeline closed."
        )