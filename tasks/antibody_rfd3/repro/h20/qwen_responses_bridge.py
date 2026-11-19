"""Local transport bridge for vLLM 0.11.2 Qwen tool outputs (Chat Completions to Responses SSE)."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import secrets
import threading
import urllib.error
import urllib.request
from uuid import uuid4

def content(value):
    if isinstance(value,str): return value
    return "\n".join(item.get("text","") for item in value or [] if item.get("type") in {"input_text","output_text","text"})

def chat_messages(req):
    messages=[]
    if req.get("instructions"): messages.append({"role":"system","content":req["instructions"]})
    for row in req["input"]:
        kind=row.get("type")
        if kind=="reasoning": continue
        if kind=="function_call":
            call={"id":row["call_id"],"type":"function","function":{"name":row["name"],"arguments":row["arguments"]}}
            if messages and messages[-1]["role"]=="assistant":
                messages[-1].setdefault("tool_calls",[]).append(call)
            else: messages.append({"role":"assistant","content":"","tool_calls":[call]})
        elif kind=="function_call_output":
            messages.append({"role":"tool","tool_call_id":row["call_id"],"content":content(row["output"])})
        else:
            role=row["role"]
            messages.append({"role":"system" if role=="developer" else role,"content":content(row["content"])})
    return messages

def start(bridge,key,url):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.headers.get("Authorization")!="Bearer "+key:
                self.send_error(403); return
            req=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            chat={"model":req["model"],"messages":chat_messages(req),"temperature":0.0,"max_tokens":2048,
                  "chat_template_kwargs":{"enable_thinking":False}}
            if req.get("tools"):
                chat["tools"]=[{"type":"function","function":{k:t[k] for k in ("name","description","parameters") if k in t}} for t in req["tools"]]
                chat["tool_choice"]=req.get("tool_choice","auto")
            try:
                with opener.open(urllib.request.Request(url+"/chat/completions",data=json.dumps(chat).encode(),
                        headers={"Content-Type":"application/json","Authorization":"Bearer "+key}),timeout=180) as res:
                    answer=json.load(res)
            except urllib.error.HTTPError as exc:
                detail=exc.read().decode()[:600]
                print("Upstream:",detail,flush=True)
                self.send_error(502); return
            choice=answer["choices"][0]["message"]
            items=[]
            if choice.get("content"):
                items.append({"type":"message","id":"msg_"+uuid4().hex,"role":"assistant","status":"completed","content":[{"type":"output_text","text":choice["content"],"annotations":[]}]})
            for call in choice.get("tool_calls") or []:
                items.append({"type":"function_call","id":"fc_"+uuid4().hex,"call_id":call["id"],
                              "name":call["function"]["name"],"arguments":call["function"]["arguments"],"status":"completed"})
            usage=answer["usage"]
            response={"id":answer["id"],"model":answer["model"],"status":"completed","output":items,
                      "usage":{"input_tokens":usage["prompt_tokens"],"output_tokens":usage["completion_tokens"],"total_tokens":usage["total_tokens"],
                      "input_tokens_details":{"cached_tokens":(usage.get("prompt_tokens_details") or {}).get("cached_tokens",0)},
                      "output_tokens_details":{"reasoning_tokens":(usage.get("completion_tokens_details") or {}).get("reasoning_tokens",0)}}}
            if req.get("stream"):
                events=[{"type":"response.created","response":dict(response,status="in_progress",output=[])}]
                for i,item in enumerate(items):
                    initial=dict(item,arguments="") if item["type"]=="function_call" else dict(item,content=[])
                    events.append({"type":"response.output_item.added","output_index":i,"item":initial})
                    if item["type"]=="function_call":
                        events.append({"type":"response.function_call_arguments.done","output_index":i,"arguments":item["arguments"]})
                    events.append({"type":"response.output_item.done","output_index":i,"item":item})
                events.append({"type":"response.completed","response":response})
                body=("".join("data: "+json.dumps(e)+"\n\n" for e in events)+"data: [DONE]\n\n").encode()
                mime="text/event-stream"
            else:
                body=json.dumps(response).encode(); mime="application/json"
            self.send_response(200);self.send_header("content-type",mime);self.send_header("content-length",str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args): pass
    server=HTTPServer((bridge,0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    return server
