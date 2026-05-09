#ifndef __SAFEQUEUE_N_
#define __SAFEQUEUE_N_

#include<iostream>
#include<queue>
#include<mutex>
#include<condition_variable>

using namespace std;

template<typename T>

class SafeQueue{
public:
    SafeQueue(size_t maxsize_in) : maxsize(maxsize_in){};    
    ~SafeQueue(){};
    
    void enqueue(const T& t)
    {
        unique_lock<mutex> lock(m);
        c.wait(lock,[this] {return q.size() < maxsize;}); 
        q.push(t);
        c.notify_one(); 
    }

    void dequeue(T& t)
    {
        unique_lock<mutex> lock(m);
        c.wait(lock,[this] {return !q.empty();}); 
        t=q.front();
        q.pop();
        c.notify_one();
    }

    size_t size() const {
    lock_guard<mutex> lock(m);
    return q.size();
    }

    bool empty()
    {
        lock_guard<mutex> lock(m);
        return q.empty();
    }

private:
    queue<T> q;
    mutable mutex m;
    condition_variable c;
    size_t maxsize;

};

#endif