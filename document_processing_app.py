from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, ContentFormat
from datetime import datetime, timezone
import os
import tempfile
from openai import AzureOpenAI
import time
import base64
from pdf2image import convert_from_path
from docx2pdf import convert
from PIL import Image
import io
import imghdr

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['document_intelligence_endpoint'] = request.form['document_intelligence_endpoint']
        session['document_intelligence_api_key'] = request.form['document_intelligence_api_key']
        session['openai_endpoint'] = request.form['openai_endpoint']
        session['openai_api_key'] = request.form['openai_api_key']
        return redirect(url_for('upload_file'))
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("Form data:", request.form)
        print("Files:", request.files)
        
        file = request.files['file']
        option = request.form.get('option', 'document_intelligence')
        
        print("Selected option:", option)
        print("Filename:", file.filename)
        print("MIME Type:", file.mimetype)
        
        # Get the file extension from the uploaded file
        extension = os.path.splitext(file.filename)[1]
        print("Initial extension from filename:", extension)
        
        if extension == '':
            # Try to get the extension from the MIME type
            mime_type = file.mimetype
            print("MIME type for extension detection:", mime_type)
            
            # Mapping of MIME types to file extensions
            mime_extension_map = {
                'image/jpeg': '.jpg',
                'image/png': '.png',
                'image/gif': '.gif',
                'image/bmp': '.bmp',
                'application/pdf': '.pdf',
                'application/msword': '.doc',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                'text/plain': '.txt'
            }
            
            extension = mime_extension_map.get(mime_type, '')
            print("Extension after checking MIME type:", extension)
        
        if extension == '':
            # As a last resort, try to detect image type using content
            print("Attempting to detect file extension from content...")
            file_content = file.read()
            extension = get_image_extension(io.BytesIO(file_content))
            file.stream.seek(0)  # Reset the file stream position
            if extension:
                print("Extension detected from file content:", extension)
            else:
                print("Unable to detect file extension from content.")
                return "An error occurred: Unsupported file type", 400
        
        # Create a temporary file with the correct extension
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
        temp_file_path = temp_file.name
        temp_file.close()

        try:
            file.save(temp_file_path)
            print(f"File saved to: {temp_file_path}")
            
            if option == 'document_intelligence':
                # Existing Document Intelligence logic
                document_intelligence_client = DocumentIntelligenceClient(
                    endpoint=session['document_intelligence_endpoint'],
                    credential=AzureKeyCredential(session['document_intelligence_api_key'])
                )

                di_start_time = datetime.now(timezone.utc).astimezone()
                di_start_timestamp = time.time()

                with open(temp_file_path, "rb") as document:
                    document_bytes = document.read()
                    poller = document_intelligence_client.begin_analyze_document(
                        "prebuilt-layout",
                        AnalyzeDocumentRequest(bytes_source=document_bytes),
                        output_content_format=ContentFormat.MARKDOWN,
                    )
                    result = poller.result()

                di_end_time = datetime.now(timezone.utc).astimezone()
                di_end_timestamp = time.time()

                document_content = result.content
                session['document_content'] = document_content

                client = AzureOpenAI(
                    api_key=session['openai_api_key'],
                    api_version="2024-06-01",
                    azure_endpoint=session['openai_endpoint']
                )

                llm_start_time = datetime.now(timezone.utc).astimezone()
                llm_start_timestamp = time.time()

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                        {"role": "user", "content": f"I want you to extract the information in json format,First you have to include page number as heading and then the name of the patient, address of the patient and name of clinic and address of the clinic and the name of the test ordered and any other information that you think is important: {document_content}"}
                    ]
                )
                
                # Extract token usage
                usage = response.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens

                llm_end_time = datetime.now(timezone.utc).astimezone()
                llm_end_timestamp = time.time()

                extracted_json = response.choices[0].message.content

                di_duration = di_end_timestamp - di_start_timestamp
                llm_duration = llm_end_timestamp - llm_start_timestamp
                total_duration = di_duration + llm_duration

                # return render_template('results.html', 
                #                        document_content=document_content, 
                #                        extracted_json=extracted_json,
                #                        di_start_time=di_start_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                #                        di_end_time=di_end_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                #                        di_duration=f"{di_duration:.2f}",
                #                        llm_start_time=llm_start_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                #                        llm_end_time=llm_end_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                #                        llm_duration=f"{llm_duration:.2f}",
                #                        total_duration=f"{total_duration:.2f}",
                #                        prompt_tokens=prompt_tokens,
                #                        completion_tokens=completion_tokens,
                #                        total_tokens=total_tokens)

            elif option == 'llm_direct':
                # Start time for the overall process
                process_start_time = datetime.now(timezone.utc).astimezone()
                process_start_timestamp = time.time()

                images = convert_file_to_images(temp_file_path)
                result = process_images_with_llm(images)
                extracted_json = result['content']

                # End time for the overall process
                process_end_time = datetime.now(timezone.utc).astimezone()
                process_end_timestamp = time.time()
                process_duration = process_end_timestamp - process_start_timestamp

                llm_start_time = result['llm_start_time']
                llm_end_time = result['llm_end_time']
                llm_duration = result['llm_duration']
                prompt_tokens = result['prompt_tokens']
                completion_tokens = result['completion_tokens']
                total_tokens = result['total_tokens']
                total_duration = process_duration

                # Commented out the code per instructions
                # return render_template('results.html', 
                #                        extracted_json=extracted_json,
                #                        llm_start_time=result['llm_start_time'],
                #                        llm_end_time=result['llm_end_time'],
                #                        llm_duration=result['llm_duration'],
                #                        prompt_tokens=result['prompt_tokens'],
                #                        completion_tokens=result['completion_tokens'],
                #                        total_tokens=result['total_tokens'],
                #                        process_start_time=process_start_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                #                        process_end_time=process_end_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                #                        process_duration=f"{process_duration:.2f} seconds")

            else:
                return "Invalid option selected", 400
            
            return render_template('results.html', 
                                   document_content=document_content, 
                                   extracted_json=extracted_json,
                                   di_start_time=di_start_time.strftime("%Y-%m-%d %H:%M:%S %Z") if di_start_time else None,
                                   di_end_time=di_end_time.strftime("%Y-%m-%d %H:%M:%S %Z") if di_end_time else None,
                                   di_duration=f"{di_duration:.2f}" if di_duration else None,
                                   llm_start_time=llm_start_time.strftime("%Y-%m-%d %H:%M:%S %Z") if llm_start_time else None,
                                   llm_end_time=llm_end_time.strftime("%Y-%m-%d %H:%M:%S %Z") if llm_end_time else None,
                                   llm_duration=f"{llm_duration:.2f}" if llm_duration else None,
                                   total_duration=f"{total_duration:.2f}",
                                   prompt_tokens=prompt_tokens,
                                   completion_tokens=completion_tokens,
                                   total_tokens=total_tokens)

        except Exception as e:
            print(f"Error in upload_file: {str(e)}")
            return f"An error occurred: {str(e)}", 500

        finally:
            try:
                os.remove(temp_file_path)
                print(f"Temporary file removed: {temp_file_path}")
            except Exception as e:
                print(f"Failed to remove temporary file: {str(e)}")

    return render_template('upload.html')

def convert_file_to_images(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    print(f"Processing file with extension: {file_extension}")
    
    try:
        if file_extension == '.pdf':
            return convert_from_path(file_path)
        elif file_extension == '.docx':
            pdf_path = file_path.replace('.docx', '.pdf')
            convert(file_path, pdf_path)
            return convert_from_path(pdf_path)
        elif file_extension in ['.txt', '.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            img = Image.open(file_path)
            print(f"Image opened successfully: {img.format}, {img.size}, {img.mode}")
            return [img]
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        print(f"Error in convert_file_to_images: {str(e)}")
        raise

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def process_images_with_llm(images):
    client = AzureOpenAI(
        api_key=session['openai_api_key'],
        api_version="2024-06-01",
        azure_endpoint=session['openai_endpoint']
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that helps people find information from images. "
                "Extract all relevant information and format it as JSON. "
                "Always provide the extracted information in English, regardless of the language in the image."
            )
        }
    ]

    for i, img in enumerate(images):
        base64_image = encode_image(img)
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Extract all relevant information from this image (page {i+1}) and add it to the JSON. "
                        "Ensure all extracted text and information is in English."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        })

    # Record start time
    llm_start_time = datetime.now(timezone.utc).astimezone()
    llm_start_timestamp = time.time()

    # Call the LLM
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500
        )

        # Extract token usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

    except Exception as e:
        raise Exception(f"Error calling Azure OpenAI API: {str(e)}")

    # Record end time
    llm_end_time = datetime.now(timezone.utc).astimezone()
    llm_end_timestamp = time.time()

    # Calculate duration
    llm_duration = llm_end_timestamp - llm_start_timestamp

    # Return extracted content and metadata
    return {
        "content": response.choices[0].message.content,
        "llm_start_time": llm_start_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "llm_end_time": llm_end_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "llm_duration": f"{llm_duration:.2f} seconds",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }
    

def get_image_extension(file_obj):
    """
    Detects the image file extension from the file content.
    """
    image_type = imghdr.what(file_obj)
    if image_type:
        return f'.{image_type}'
    return ''

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['question']
    document_content = session.get('document_content', '')
    extracted_json = session.get('extracted_json', '')

    client = AzureOpenAI(
        api_key=session['openai_api_key'],
        api_version="2024-06-01",
        azure_endpoint=session['openai_endpoint']
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question based on the given document content or extracted information. Answer always in english."},
            {"role": "user", "content": f"Document content: {document_content}\nExtracted information: {extracted_json}\n\nQuestion: {question}"}
        ]
    )

    answer = response.choices[0].message.content
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)