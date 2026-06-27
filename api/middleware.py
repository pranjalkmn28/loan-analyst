class CORSMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method == "OPTIONS":
            from django.http import HttpResponse
            r = HttpResponse()
            self._set_headers(r)
            return r
        response = self.get_response(request)
        self._set_headers(response)
        return response

    def _set_headers(self, response):
        response["Access-Control-Allow-Origin"]  = "*"
        response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"