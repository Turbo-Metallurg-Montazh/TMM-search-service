from fastapi import HTTPException
from app.services.catalogs.catalogsDelete import DeleteCatalogsRequest, validate_delete_catalog_request, delete_catalog_files

import app.state as state

def delete_many_catalogs_service(request: DeleteCatalogsRequest):
    try:
        filenames = []

        for item in request.files:
            filename = item.filename

            if not filename.endswith(f".{item.extension}"):
                filename = f"{filename}.{item.extension}"

            filenames.append(filename)

        validate_delete_catalog_request(filenames)

        deleted = delete_catalog_files(filenames)

        state.mark_index_stale()

        return {
            "Status": "ok",
            "deletedCount": len(deleted),
            "deletedFiles": deleted,
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {exc}"
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )