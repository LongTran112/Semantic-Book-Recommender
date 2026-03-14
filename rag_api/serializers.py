"""DRF serializers for RAG API payload contracts."""

from __future__ import annotations

from rest_framework import serializers


class RagFiltersSerializer(serializers.Serializer):
    categories = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_null=True,
        default=None,
    )
    learning_modes = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        allow_null=True,
        default=None,
    )
    min_similarity = serializers.FloatField(required=False, default=-1.0)


class RetrievalConfigSerializer(serializers.Serializer):
    hybrid_enabled = serializers.BooleanField(required=False, default=True)
    dense_weight = serializers.FloatField(required=False, default=0.7)
    lexical_weight = serializers.FloatField(required=False, default=0.3)
    image_weight = serializers.FloatField(required=False, default=0.2)
    candidate_pool_size = serializers.IntegerField(required=False, default=48, min_value=1)
    final_top_k = serializers.IntegerField(required=False, default=8, min_value=1)
    reranker_enabled = serializers.BooleanField(required=False, default=True)
    reranker_model_name = serializers.CharField(required=False, allow_null=True, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_top_n = serializers.IntegerField(required=False, default=32, min_value=1)
    modalities = serializers.ListField(
        child=serializers.ChoiceField(choices=["text", "image"]),
        required=False,
        default=list,
    )
    query_image_path = serializers.CharField(required=False, default="", allow_blank=True)
    visual_model_tag = serializers.CharField(required=False, default="", allow_blank=True)


class LlamaCppConfigSerializer(serializers.Serializer):
    enabled = serializers.BooleanField(required=False, default=False)
    model_path = serializers.CharField(required=False, default="", allow_blank=True)
    n_ctx = serializers.IntegerField(required=False, default=2048)
    max_tokens = serializers.IntegerField(required=False, default=420)
    temperature = serializers.FloatField(required=False, default=0.2)
    top_p = serializers.FloatField(required=False, default=0.9)
    n_threads = serializers.IntegerField(required=False, default=6)
    n_gpu_layers = serializers.IntegerField(required=False, default=0)
    seed = serializers.IntegerField(required=False, default=42)


class OllamaConfigSerializer(serializers.Serializer):
    enabled = serializers.BooleanField(required=False, default=False)
    base_url = serializers.CharField(required=False, default="http://127.0.0.1:11434")
    model = serializers.CharField(required=False, default="granite3.3:8b")
    temperature = serializers.FloatField(required=False, default=0.2)
    top_p = serializers.FloatField(required=False, default=0.9)
    num_ctx = serializers.IntegerField(required=False, default=8192)
    timeout_sec = serializers.IntegerField(required=False, default=180)


class ImageGenerationConfigSerializer(serializers.Serializer):
    enabled = serializers.BooleanField(required=False, default=False)
    provider = serializers.CharField(required=False, default="none")
    endpoint_url = serializers.CharField(required=False, default="http://127.0.0.1:7860/sdapi/v1/txt2img")
    output_dir = serializers.CharField(required=False, default="output/generated_images")
    num_images = serializers.IntegerField(required=False, default=1, min_value=1)
    width = serializers.IntegerField(required=False, default=768, min_value=128)
    height = serializers.IntegerField(required=False, default=768, min_value=128)
    guidance_scale = serializers.FloatField(required=False, default=7.0)
    steps = serializers.IntegerField(required=False, default=25, min_value=1)
    negative_prompt = serializers.CharField(required=False, default="", allow_blank=True)
    prompt_suffix = serializers.CharField(required=False, default="clean diagram style, high detail", allow_blank=True)
    timeout_sec = serializers.IntegerField(required=False, default=120, min_value=5)


class RagRequestSerializer(serializers.Serializer):
    query = serializers.CharField(min_length=1)
    top_k = serializers.IntegerField(required=False, default=8, min_value=1)
    max_citations = serializers.IntegerField(required=False, default=6, min_value=1)
    allow_fallback = serializers.BooleanField(required=False, default=True)
    filters = RagFiltersSerializer(required=False, default=dict)
    retrieval = RetrievalConfigSerializer(required=False, default=dict)
    llm = LlamaCppConfigSerializer(required=False, default=dict)
    ollama = OllamaConfigSerializer(required=False, default=dict)
    image_generation = ImageGenerationConfigSerializer(required=False, default=dict)

    def validate_query(self, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise serializers.ValidationError("This field may not be blank.")
        return clean
