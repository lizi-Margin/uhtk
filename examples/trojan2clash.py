# trojan
# 说明：读取 trojan_lines.txt（每行一个 trojan://...），输出 proxies.yaml（YAML 片段）

import sys
from urllib.parse import urlparse, unquote, parse_qs

INPUT = "temp.txt"
DECODE = True
OUTPUT = "proxies.yaml"

def bool_from(val):
    if val is None:
        return False
    s = str(val).lower()
    return s in ("1","true","yes","on")

def decode_base64(stuff: "str"):
    import base64
    decoded_bytes = base64.b64decode(stuff)
    decoded_str = decoded_bytes.decode("utf-8")
    return decoded_str

def parse_trojan(uri):
    # urlparse 能解析出 username/password 部分（trojan 常把 password 放在 username）
    p = urlparse(uri.strip())
    if p.scheme != "trojan":
        return None
    # p.username is the "password" in trojan://password@host:port
    password = p.username or ""
    server = p.hostname or ""
    port = p.port or 443
    fragment = unquote(p.fragment) if p.fragment else ""
    q = parse_qs(p.query)
    # 常见参数：sni, allowInsecure, network, path, peer, udp
    sni = q.get("sni", [None])[0]
    allow_insecure = q.get("allowInsecure", [None])[0]
    network = q.get("network", [None])[0]  # e.g. ws, grpc, tcp
    path = q.get("path", [None])[0]
    peer = q.get("peer", [None])[0]

    name = fragment or f"trojan-{server}-{port}"
    entry = {
        "name": name,
        "type": "trojan",
        "server": server,
        "port": port,
        "password": password,
        # optional fields
        "sni": sni,
        "skip-cert-verify": bool_from(allow_insecure),
        "network": network,
        "ws-opts": None,
        "grpc-opts": None,
    }

    # 如果 network=ws，尝试填 ws-opts.path / headers.Host (use peer or sni)
    if network == "ws":
        ws_opts = {}
        if path:
            ws_opts["path"] = path
        if peer:
            ws_opts.setdefault("headers", {})["Host"] = peer
        elif sni:
            ws_opts.setdefault("headers", {})["Host"] = sni
        entry["ws-opts"] = ws_opts or None

    # 如果 network=grpc，可以填 grpc-opts.service-name 从 path 或 peer (有时为服务名)
    if network == "grpc":
        grpc_opts = {}
        if path:
            grpc_opts["grpc-service-name"] = path.lstrip("/")
        entry["grpc-opts"] = grpc_opts or None

    return entry

def to_yaml_entries(entries):
    lines = ["proxies:"]
    for e in entries:
        lines.append("  - name: \"{}\"".format(e["name"].replace('"','\\"')))
        lines.append("    type: {}".format(e["type"]))
        lines.append("    server: {}".format(e["server"]))
        lines.append("    port: {}".format(e["port"]))
        lines.append("    password: \"{}\"".format(e["password"].replace('"','\\"')))
        if e.get("sni"):
            lines.append("    sni: {}".format(e["sni"]))
        if e.get("skip-cert-verify"):
            lines.append("    skip-cert-verify: true")
        if e.get("network"):
            lines.append("    network: {}".format(e["network"]))
            if e["network"] == "ws" and e.get("ws-opts"):
                lines.append("    ws-opts:")
                if e["ws-opts"].get("path"):
                    lines.append("      path: {}".format(e["ws-opts"]["path"]))
                if e["ws-opts"].get("headers"):
                    lines.append("      headers:")
                    for k,v in e["ws-opts"]["headers"].items():
                        lines.append(f"        {k}: {v}")
            if e["network"] == "grpc" and e.get("grpc-opts"):
                lines.append("    grpc-opts:")
                for k,v in e["grpc-opts"].items():
                    lines.append(f"      {k}: {v}")
        lines.append("")  # blank line between proxies
    return "\n".join(lines)

def main():
    if DECODE:
        with open(INPUT, "r", encoding="utf-8") as f_base64:
            decoded = decode_base64(f_base64.read())
        lines = decoded.splitlines()
    else:
        with open(INPUT, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    # print(lines)
    uris = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    entries = []
    for u in uris:
        e = parse_trojan(u)
        if e:
            entries.append(e)
        else:
            print("警告：无法解析 URI，跳过：", u, file=sys.stderr)
    if not entries:
        print("未找到有效 trojan URI。", file=sys.stderr)
        return
    yaml_text = to_yaml_entries(entries)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    print(f"已生成 {OUTPUT}，请将其中的 proxies 部分合并到 Clash 主配置的 proxies: 下。")

if __name__ == "__main__":
    main()
