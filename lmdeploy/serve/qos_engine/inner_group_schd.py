import queue
import threading
import logging
logger = logging.getLogger(__name__)

class UserRequestQueue:
    """
    Inner group user request queues
    """

    def __init__(self, group: str, user_id_map: dict):
        self.group = group
        self.user_queue_map = dict()
        self.user_quota_map = dict()
        self.user_id_maps = user_id_map

        total_quota = 0
        for item in user_id_map:
            total_quota += item["quota_pct"]
        for item in user_id_map:
            user_id = item["id"]
            self.user_queue_map[user_id] = queue.Queue()
            self.user_quota_map[user_id] = item["quota_pct"] / total_quota

        self.lock = threading.Lock()

    def enqueue(self, request_event):
        """
        Enqueue request to correspoding user queue.
        """
        if request_event[0].user_id in self.user_queue_map:
            self.user_queue_map[request_event[0].user_id].put(request_event)
        else:
            self.user_queue_map["default"].put(request_event)
    
    def empty(self):
        """
        Whether all user queues are empty.
        """
        with self.lock:
            for _, user_queue in self.user_queue_map.items():
                if not user_queue.empty():
                    return False
        return True
    
    def dequeue(self, usage_stats):
        """
        Dequeue the request to serve.
        """
        with self.lock:
            uid_to_serve = self.user_to_serve(usage_stats)
            if uid_to_serve in self.user_queue_map:
                return self.user_queue_map[uid_to_serve].get()
        
        return None

    def user_to_serve(self, usage_stats):
        """
        Inner group scheduling.
        Find the user to serve from user request queues.
        """
        min_usage = 100
        uid_to_serve = ""
        for uid, req_queue in self.user_queue_map.items():
            if req_queue.empty():
                continue

            # TODO: include token length
            # Calculate current user's actual used share and quota share
            user_usage, _, group_usage, _ = usage_stats.get_user_usage(uid, self.group)
            actual_share = (user_usage / group_usage) if group_usage > 0 else 0
            due_share = self.user_quota_map[uid]

            # Serve the user with the relatively least usage share
            curr_usage = actual_share / due_share
            if curr_usage == 0:
                return uid
            if curr_usage < min_usage:
                uid_to_serve = uid
                min_usage = curr_usage
        return uid_to_serve
