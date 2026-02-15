from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os

load_dotenv()

# # Initialize ChatOpenAI
# llm = ChatOpenAI(
#     model="gpt-3.5-turbo",  # Using gpt-3.5-turbo as it's widely available
#     api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0.7
# )

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    deployment_name="gpt4-fast",  # IMPORTANT
    temperature=0.7
)

response = llm.invoke("Hello")

print(response.content)

# Test 1: Simple query
print("=" * 50)
print("TEST 1: Simple Query")
print("=" * 50)

try:
    response = llm.invoke("What is the capital of France?")
    print(f"✓ Success!\nResponse: {response.content}\n")
except Exception as e:
    print(f"✗ Error: {e}\n")

# # Test 2: STEM Explanation
# print("=" * 50)
# print("TEST 2: STEM Explanation")
# print("=" * 50)

# try:
#     response = llm.invoke("Explain photosynthesis in simple terms for a high school student.")
#     print(f"✓ Success!\nResponse: {response.content}\n")
# except Exception as e:
#     print(f"✗ Error: {e}\n")

# # Test 3: Math Problem
# print("=" * 50)
# print("TEST 3: Math Problem")
# print("=" * 50)

# try:
#     response = llm.invoke("Solve: 5x + 10 = 35. Show step by step.")
#     print(f"✓ Success!\nResponse: {response.content}\n")
# except Exception as e:
#     print(f"✗ Error: {e}\n")

# # Test 4: Stream response (optional)
# print("=" * 50)
# print("TEST 4: Streaming Response")
# print("=" * 50)

# try:
#     print("Response (streaming): ", end="", flush=True)
#     for chunk in llm.stream("What is quantum physics?"):
#         print(chunk.content, end="", flush=True)
#     print("\n✓ Success!\n")
# except Exception as e:
#     print(f"\n✗ Error: {e}\n")

# print("=" * 50)
# print("All tests completed!")
# print("=" * 50)
