from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableLambda
from utility import model_instances as mi
from langchain_core.output_parsers import JsonOutputParser
from llama_cpp import LlamaGrammar

if __name__ == "__main__":
    llm = mi.get_llama_instance()
    grammar = LlamaGrammar.from_file("./grammar2.gbnf")
    # 定义提示模板
    keyword_prompt = PromptTemplate(
        input_variables=["text"],
        template="hi, I am {text}",
    )

    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="{text}"
    )

    def debug_step_func(inputs):
        print(f"Debug: {inputs}")

    debug_step = RunnableLambda(
        func=lambda inputs: (print(f"Debug: {inputs}"), inputs)[1]  # 打印并返回原数据
    )

    # 定义可调整调用参数的 LLM 步骤
    runnable_binding = llm.bind(stop=['\n'])

    # 创建流水线
    pipeline = keyword_prompt | debug_step | runnable_binding | debug_step | summary_prompt | debug_step | runnable_binding


    # 测试流水线
    result = pipeline.invoke({"text": "Mike"}, verbose = True)
    print(result)