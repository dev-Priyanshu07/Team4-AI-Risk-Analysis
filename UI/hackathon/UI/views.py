from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Q  # For OR conditions in DB queries
import json
import os
import re
import uuid
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from .models import Transaction
# Load Home Page
def index(request):
    return render(request, "index.html")

#search for the entity
@csrf_exempt
def search_entity(request):
    """Search for all records related to an entity (Sender or Receiver)."""
    if request.method == 'GET':
        entity_name = request.GET.get('entity', '').strip()
        if not entity_name:
            return JsonResponse({'error': 'Entity name required'}, status=400)

        # Search for entity in sender or receiver fields
        transactions = Transaction.objects.filter(
            Q(sender__icontains=entity_name) | Q(receiver__icontains=entity_name)
        )

        if not transactions.exists():
            return JsonResponse({'message': 'No records found'}, status=404)

        results = []
        for txn in transactions:
            results.append({
                "Transaction ID": str(txn.transaction_id),
                "Sender": txn.sender if txn.sender else "Unknown",
                "Receiver": txn.receiver if txn.receiver else "Unknown",
                "Entity Type": txn.entity_type,
                "Risk Score": txn.risk_score,
                "Supporting Evidence": txn.supporting_evidence,
                "Confidence Score": txn.confidence_score,
                "Reason": txn.reason,
                "Full Text": txn.full_text
            })

        return JsonResponse({"results": results}, status=200)

    return JsonResponse({'error': 'Invalid request'}, status=400)


#upload the file
@csrf_exempt
def handle_upload(request, folder="structured"):
    """Uploads, processes structured & unstructured data, saves to DB, and returns extracted transactions."""
    if request.method == 'POST' and request.FILES.get('file'):
        upload_dir = os.path.join(settings.MEDIA_ROOT, folder)
        os.makedirs(upload_dir, exist_ok=True)  # Ensure directory exists

        uploaded_file = request.FILES['file']
        file_path = os.path.join(upload_dir, uploaded_file.name)

        # Save file
        with default_storage.open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Process File
        extracted_transactions = process_file(file_path, folder)
        if not extracted_transactions:
            return JsonResponse({'error': 'No transactions found in file'}, status=400)

        # Save results to the database
        results = []
        for txn in extracted_transactions:
            transaction = Transaction.objects.create(
                sender=txn.get('sender', None),
                receiver=txn.get('receiver', None),
                entity_type=txn.get('entity_type', ["Unknown"]),
                risk_score=txn.get('risk_score', 0.5),  # Default risk score
                supporting_evidence=txn.get('supporting_evidence', ["No evidence available"]),
                confidence_score=txn.get('confidence_score', 0.5),  # Default confidence
                reason=txn.get('reason', "No risk detected."),
                full_text=txn['full_text']
            )
            results.append({
                "Transaction ID": str(transaction.transaction_id),  # UUID format
                "Sender": transaction.sender,
                "Receiver": transaction.receiver,
                "Entity Type": transaction.entity_type,
                "Risk Score": transaction.risk_score,
                "Supporting Evidence": transaction.supporting_evidence,
                "Confidence Score": transaction.confidence_score,
                "Reason": transaction.reason,
            })

        return JsonResponse({"results": results}, status=200)

    return JsonResponse({'error': 'Upload failed'}, status=400)


#process the file
def process_file(file_path, folder):
    """Process structured and unstructured files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return []

    transactions = content.split('---')
    extracted_data = []

    for transaction in transactions:
        sender_match = re.search(r"Sender:\s*- Name:\s*\"(.*?)\"", transaction)
        receiver_match = re.search(r"Receiver:\s*- Name:\s*\"(.*?)\"", transaction)

        if folder == "structured" and sender_match and receiver_match:
            sender_name = sender_match.group(1)
            receiver_name = receiver_match.group(1)
            extracted_data.append({
                "sender": sender_name,
                "receiver": receiver_name,
                "entity_type": ["Corporation"],  # Default structured entity
                "risk_score": 0.9,
                "supporting_evidence": ["Company database"],
                "confidence_score": 0.85,
                "reason": f"{sender_name} and {receiver_name} appear in company records.",
                "full_text": transaction.strip()
            })
        else:
            # Handle unstructured data
            extracted_data.append({
                "sender": None,
                "receiver": None,
                "entity_type": ["Unknown"],  # Unstructured entity
                "risk_score": 0.5,  # Lower risk score for unknown
                "supporting_evidence": ["No clear match"],
                "confidence_score": 0.5,
                "reason": "Could not extract structured sender/receiver data.",
                "full_text": transaction.strip()
            })

    return extracted_data

@csrf_exempt
def search_entity(request):
    """Search for all records related to an entity (Sender or Receiver)."""
    if request.method == 'GET':
        entity_name = request.GET.get('entity', '').strip()
        if not entity_name:
            return JsonResponse({'error': 'Entity name required'}, status=400)

        transactions = Transaction.objects.filter(sender__icontains=entity_name) | \
                       Transaction.objects.filter(receiver__icontains=entity_name)

        if not transactions.exists():
            return JsonResponse({'message': 'No records found'}, status=404)

        results = [{"Transaction ID": str(txn.transaction_id), "Sender": txn.sender, "Receiver": txn.receiver} for txn in transactions]

        return JsonResponse({"results": results}, status=200)

    return JsonResponse({'error': 'Invalid request'}, status=400)
