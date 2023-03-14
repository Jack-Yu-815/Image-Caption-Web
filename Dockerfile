FROM public.ecr.aws/lambda/python:3.9.2022.12.30.19

# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY eval.py ${LAMBDA_TASK_ROOT}
COPY models.py ${LAMBDA_TASK_ROOT}
COPY utils.py ${LAMBDA_TASK_ROOT}
COPY tokenizer.pickle ${LAMBDA_TASK_ROOT}
COPY 8K_caption_24.torch ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.lambda_handler" ]
