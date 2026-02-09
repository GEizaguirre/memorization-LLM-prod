from enum import Enum

DEFAULT_TEMPERATURE = 0
INITIAL_TEXT_TOKENS = 50
TOKENS_PER_WORD = 1.35
MAX_ITERATIONS = 20
MAXIMUM_PHASE1_TOKENS = 1000
MAXIMUM_OUTPUT_TOKENS = 1000
PHASE1_SUCCESS_THRESHOLD = 0.6
MAX_BEST_OF_N = 100


class Provider(str, Enum):
    GOOGLE = "google"
    CLAUDE = "claude"
    OPENAI = "openai"
    MOONSHOT = "moonshot"
    DEEPSEEK = "deepseek"


class Model(str, Enum):
    CLAUDE_OPUS_4_5 = "claude-opus-4-5"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5"
    GPT_5_2 = "gpt-5.2"
    GPT_5 = "gpt-5"
    GPT_4O = "gpt-4o"
    KIMI_K2_5 = "kimi-k2.5"
    DEEPSEEK_CHAT = "deepseek-chat"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"
    GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"


MODEL_TO_PROVIDER = {
    Model.CLAUDE_OPUS_4_5: Provider.CLAUDE,
    Model.CLAUDE_SONNET_4_5: Provider.CLAUDE,
    Model.GPT_5_2: Provider.OPENAI,
    Model.GPT_5: Provider.OPENAI,
    Model.GPT_4O: Provider.OPENAI,
    Model.KIMI_K2_5: Provider.MOONSHOT,
    Model.DEEPSEEK_CHAT: Provider.DEEPSEEK,
    Model.GEMINI_2_5_FLASH: Provider.GOOGLE,
    Model.GEMINI_2_5_PRO: Provider.GOOGLE,
    Model.GEMINI_3_PRO_PREVIEW: Provider.GOOGLE,
    Model.GEMINI_3_FLASH_PREVIEW: Provider.GOOGLE,
}
