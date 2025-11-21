import os
import fitz
import base64
import logging

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


load_dotenv()
logger = logging.getLogger(__name__)


class VLLMNotInitialized(BaseException): ...


class PDFParser:

    def __init__(self):
        """
        Initialize PDFParser with VLLM configuration from environment variables.

        Environment Variables:
            GROQ_URL: GroQ API URL (default: https://api.groq.com/openai/v1)
            GROQ_API_KEY: API key for GroQ (tried first)
            GROQ_MODELS: Comma-separated list of GroQ models
            GOOGLE_API_KEY: API key for Google Gemini (fallback #1)
            GEMINI_VLLM_MODEL_NAME: Gemini model name (default: gemini-2.5-flash)
            OPENAI_API_KEY: API key for OpenAI (fallback #2)
            OPENAI_VLLM_MODEL_NAME: OpenAI model name (default: gpt-4o)
            IMAGES_TO_PROCESS_VLLM: Maximum number of images to process (default: 3)
        """
        groq_free_vllm_models = [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
        ]

        # envs for groq free models
        self.groq_api_url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_models = os.getenv("GROQ_VLLM_MODELS", groq_free_vllm_models)
        logger.info(f"{self.groq_models=}, groq_api_key_configured={bool(self.groq_api_key)}")
        if isinstance(self.groq_models, str):
            try:
                self.groq_models = self.groq_models.split(",")
            except BaseException as e:
                logger.error(
                    f"Couldn't parse GROQ_VLLM_MODELS from envs, using defaults. Error:\n{e}"
                )
                self.groq_models = groq_free_vllm_models

        # envs for google gemini
        self.gemini_vllm_api_key = os.getenv("GOOGLE_API_KEY")
        self.gemini_vllm_model_name = os.getenv(
            "GEMINI_VLLM_MODEL_NAME", "gemini-2.5-flash"
        )

        # envs for google openai
        self.openai_vllm_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_vllm_model_name = os.getenv("OPENAI_VLLM_MODEL_NAME", "gpt-4o")

        # images to process with vllms
        self.images_to_process = int(os.getenv("IMAGES_TO_PROCESS_VLLM", 3))

    def _initialized(self):
        """
        Check if at least one VLLM API key is configured.

        Raises:
            VLLMNotInitialized: If none of the API keys are set in environment
        """
        if not any(
            [
                self.groq_api_key,
                self.gemini_vllm_api_key,
                self.openai_vllm_api_key,
            ]
        ):
            raise VLLMNotInitialized(
                "No VLLM API keys found! Please set at least one of: "
                "GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY"
            )

    def process(self, base64_content: str) -> str:
        """
        Process PDF file and extract text content.

        First attempts to extract text directly from PDF. If PDF contains
        only images or text extraction fails, uses VLLM to extract text
        from PDF page images.

        Args:
            base64_content (str): PDF file content encoded in base64

        Returns:
            str: Extracted text content from PDF
        """
        self._initialized()

        # first trying to parse pdf using python libs
        extracted_text = self._read_pdf_as_text(base64_content)
        logger.info(f"{extracted_text=}")
        if extracted_text:
            return extracted_text

        # if contains images, using vllm
        logger.info("PDF contains images, trying to extract with vllm...")
        images = self._pdf_bytes_to_images(base64.b64decode(base64_content))
        logger.info(f"{len(images)=}")
        texts = self._vllm_img2text(images)
        logger.info(f"{len(texts)=}")
        return "\n".join(texts)

    def _read_pdf_as_text(self, base64_content: str) -> str:
        """
        Reading pdf file

        Args:
            base64_content  (str): pdf content in base64

        Returns:
            Human readable PDF content
        """
        pdf_bytes = base64.b64decode(base64_content)
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        content = []
        for page in pdf_doc:
            content += [page.get_text()]

        pdf_doc.close()
        return "\n".join(content).strip()

    def _pdf_bytes_to_images(self, pdf_bytes: bytes) -> list[str]:
        """
        Convert PDF pages to base64-encoded PNG images.

        Args:
            pdf_bytes (bytes): PDF file content as bytes

        Returns:
            list[str]: List of base64-encoded PNG images, one per page
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images: list[str] = []
        try:
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # medium quality
                img_bytes = pix.tobytes("png")
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                images.append(img_base64)
        finally:
            doc.close()
        return images

    def _vllm_img2text(self, images: list[str]) -> list[str]:
        """
        Extract text from images using Vision Language Model.

        Pipeline:
        1. Try GroQ models (all models in the list)
        2. If all fail, try Google Gemini (if initialized)
        3. If Gemini fails, try OpenAI (if initialized)

        Args:
            images (list[str]): List of base64-encoded images

        Returns:
            list[str]: Extracted text from each image
        """
        texts: list[str] = []
        images_to_extract = images[: self.images_to_process]

        # Try GroQ models first
        if self.groq_api_key and self.groq_models:
            logger.info(f"Trying GroQ with {len(self.groq_models)} models...")
            for model_name in self.groq_models:
                logger.info(f"Attempting with GroQ model: {model_name}")
                llm = ChatOpenAI(
                    base_url=self.groq_api_url,
                    api_key=self.groq_api_key,
                    model=model_name,
                )

                success = self._extract_with_llm(llm, images_to_extract, texts)
                if success:
                    return texts
                logger.warning(
                    f"GroQ model {model_name} failed after processing {len(texts)}/{len(images_to_extract)} images, trying next..."
                )

            logger.error(
                f"All GroQ models failed. Progress: {len(texts)}/{len(images_to_extract)} images"
            )

        # Try Google Gemini if GroQ failed
        if self.gemini_vllm_api_key and len(texts) < len(images_to_extract):
            logger.info(f"Trying Google Gemini: {self.gemini_vllm_model_name}")
            llm = ChatGoogleGenerativeAI(
                google_api_key=self.gemini_vllm_api_key,
                model=self.gemini_vllm_model_name,
            )

            success = self._extract_with_llm(llm, images_to_extract, texts)
            if success:
                return texts
            logger.error(
                f"Google Gemini failed. Progress: {len(texts)}/{len(images_to_extract)} images"
            )

        # Try OpenAI as last resort
        if self.openai_vllm_api_key and len(texts) < len(images_to_extract):
            logger.info(f"Trying OpenAI: {self.openai_vllm_model_name}")
            llm = ChatOpenAI(
                api_key=self.openai_vllm_api_key, model=self.openai_vllm_model_name
            )

            success = self._extract_with_llm(llm, images_to_extract, texts)
            if success:
                return texts
            logger.error(
                f"OpenAI failed. Progress: {len(texts)}/{len(images_to_extract)} images"
            )

        logger.error(
            f"All VLLM providers failed. Final progress: {len(texts)}/{len(images_to_extract)} images"
        )
        return texts

    def _extract_with_llm(
        self,
        llm: ChatOpenAI | ChatGoogleGenerativeAI,
        images: list[str],
        texts: list[str],
    ) -> bool:
        """
        Extract text from images using provided LLM.

        Continues from where the previous LLM left off by processing only
        remaining images (starting from len(texts) index).

        Args:
            llm: Language model instance
            images (list[str]): List of base64-encoded images to process
            texts (list[str]): List to append extracted texts to (preserves progress)

        Returns:
            bool: True if extraction succeeded for all remaining images, False otherwise
        """
        start_index = len(texts)
        total_images = len(images)

        if start_index >= total_images:
            logger.info("All images already processed")
            return True

        try:
            for i in range(start_index, total_images):
                image = images[i]
                message = HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image}"},
                        },
                        {"type": "text", "text": "Extract text from image"},
                    ]
                )

                logger.info(
                    f"Processing image {i+1}/{total_images} with {llm.model_name}..."
                )
                response = llm.invoke([message])
                texts.append(response.content)

            logger.info(f"Successfully extracted text from all {len(texts)} images")
            return True

        except BaseException as e:
            logger.error(
                f"Error during extraction with {llm.__class__.__name__} at image {len(texts)+1}/{total_images}: {e}"
            )
            return False


pdf_parser_tool = PDFParser()
