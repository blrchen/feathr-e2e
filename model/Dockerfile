FROM pytorch/pytorch

COPY . /app
RUN pip install -r /app/requirements.txt

WORKDIR /app
EXPOSE 8000
CMD ["/opt/conda/bin/uvicorn", "--host", "0.0.0.0", "--port", "8000", "serve:app"]
