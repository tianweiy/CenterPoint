def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + str(k))
        else:
            flatted[start + sep + str(k)] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, str(k))
        else:
            flatted[str(k)] = v
    return flatted


def metric_to_str(metrics, sep="."):
    flatted_metrics = flat_nested_json_dict(metrics, sep)
    metrics_str_list = []
    for k, v in flatted_metrics.items():
        if isinstance(v, float):
            metrics_str_list.append(f"{k}={v:.4}")
        elif isinstance(v, (list, tuple)):
            if v and isinstance(v[0], float):
                v_str = ", ".join([f"{e:.4}" for e in v])
                metrics_str_list.append(f"{k}=[{v_str}]")
            else:
                metrics_str_list.append(f"{k}={v}")
        else:
            metrics_str_list.append(f"{k}={v}")
    return ", ".join(metrics_str_list)
