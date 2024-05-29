# CSV Chat API Documentation

## Overview

The CSV Chat API allows users to upload a CSV file and interact with it through a chat interface. It uses language models to interpret user queries and respond based on the content of the uploaded CSV file.

## Base URL
http://your-domain.com/
## Endpoints

### 1. Upload CSV

#### `POST /upload_csv/`

Upload a CSV file to the server. The uploaded file is stored and can be used in subsequent chat interactions. Only one file can be stored at a time; uploading a new file will overwrite the previous one.

**Request:**

- **Headers:**
  - `Content-Type: multipart/form-data`
- **Body:**
  - `file`: The CSV file to upload.

**Response:**

- **200 OK**: File uploaded successfully.
  ```json
  {
    "message": "File uploaded successfully"
  }

500 Internal Server Error: An error occurred during file upload
{
  "detail": "Error message"
}
Example cURL Request:
curl -X POST "http://your-domain.com/upload_csv/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path_to_your_file.csv"